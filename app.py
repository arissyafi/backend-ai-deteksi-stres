import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import mne
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import gc

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# --- KONFIGURASI ---
MODEL_FILE = 'OTAK_AI_100FILE_FINAL.pkl'
ALLOWED_EXTENSIONS = {'edf'}
TARGET_CHANNELS = ['AF3', 'T7', 'Pz', 'T8', 'AF4'] # Channel Wajib

model_data = None

# Load model saat startup
if os.path.exists(MODEL_FILE):
    try:
        model_data = joblib.load(MODEL_FILE)
        print("✅ Model berhasil dimuat.")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
else:
    print(f"⚠️ Peringatan: File {MODEL_FILE} tidak ditemukan.")

# --- FUNGSI BANTUAN ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_channel_names(raw):
    """
    Fungsi pintar untuk menebak nama channel yang agak beda.
    Misal: 'EEG AF3' -> 'AF3', 'af-3' -> 'AF3'
    """
    mapping = {}
    for ch in raw.ch_names:
        clean_name = ch.upper().replace('EEG', '').replace('-', '').strip()
        # Mapping nama asli ke nama bersih jika cocok dengan target
        if clean_name in TARGET_CHANNELS:
            mapping[ch] = clean_name
    
    if mapping:
        raw.rename_channels(mapping)
    return raw

def hjorth_params(data):
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    var_zero = np.var(data)
    var_d1 = np.var(diff1)
    if var_zero == 0: return 0, 0
    mobility = np.sqrt(var_d1 / var_zero)
    if var_d1 == 0: return 0, 0
    mobility_d1 = np.sqrt(np.var(diff2) / var_d1)
    if mobility == 0: return 0, 0
    complexity = mobility_d1 / mobility
    return mobility, complexity

def extract_features_complete(data, fs=128):
    features = []
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
             'Beta': (13, 30), 'Gamma': (30, 45)}
    
    for channel_data in data:
        features.extend([np.mean(channel_data), np.std(channel_data), 
                         skew(channel_data), kurtosis(channel_data)])
        f_mob, f_comp = hjorth_params(channel_data)
        features.extend([f_mob, f_comp])
        freqs, psd = welch(channel_data, fs, nperseg=fs)
        for band, (low, high) in bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            if np.sum(idx_band) > 0:
                features.append(np.mean(psd[idx_band]))
            else:
                features.append(0)
    return np.array(features)

@app.route('/', methods=['GET'])
def home():
    return "NeuroSense API v3.0 (Secure & Flexible)"

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Cek Model
    if model_data is None:
        return jsonify({'error': 'Server Error: Model AI belum siap.'}), 500

    # 2. Cek File Ada/Tidak
    if 'file' not in request.files:
        return jsonify({'error': 'Harap upload file.'}), 400
    
    file = request.files['file']
    
    # 3. Cek Nama File & Ekstensi
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong.'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Format salah! Hanya menerima file .edf'}), 400

    # Simpan sementara
    try:
        temp_dir = '/tmp'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': f"Gagal upload: {str(e)}"}), 500

    try:
        knn = model_data['model_knn']
        scaler = model_data['scaler']
        idx_fitur = model_data['fitur_idx']

        # --- LANGKAH BACA & VALIDASI CHANNEL ---
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        except Exception:
            raise ValueError("File rusak atau bukan format EDF valid.")

        # Normalisasi nama channel (Handling nama aneh)
        raw = normalize_channel_names(raw)
        
        # Cek Channel yang Ada
        available = raw.ch_names
        
        # Logika Fleksibilitas:
        # Cari mana channel target yang ada di file ini
        found_channels = [ch for ch in TARGET_CHANNELS if ch in available]
        missing_channels = list(set(TARGET_CHANNELS) - set(found_channels))

        # JIKA KURANG DARI 5: Tolak dengan pesan jelas
        if len(missing_channels) > 0:
            raise ValueError(f"Channel tidak lengkap. Kurang: {', '.join(missing_channels)}. Wajib ada 5 channel: AF3, T7, Pz, T8, AF4.")

        # JIKA LEBIH DARI 5: Ambil hanya yang dibutuhkan (Buang sisanya)
        raw.pick_channels(TARGET_CHANNELS)
        
        # --- PROSES STANDAR ---
        raw.filter(1.0, 45.0, fir_design='firwin', verbose=False)
        if raw.info['sfreq'] != 128: 
            raw.resample(128, verbose=False)
        
        epochs = mne.make_fixed_length_epochs(raw, duration=2.0, overlap=1.0, verbose=False)
        data_signal = epochs.get_data()

        if len(data_signal) == 0:
            raise ValueError("File valid tapi durasi terlalu pendek (min 2 detik).")

        # Ekstraksi
        fitur_mentah = []
        for i in range(len(data_signal)):
            feat = extract_features_complete(data_signal[i])
            fitur_mentah.append(feat)
        
        fitur_mentah = np.array(fitur_mentah) 

        # Scale -> Select -> Predict
        fitur_scaled = scaler.transform(fitur_mentah) 
        fitur_final = fitur_scaled[:, idx_fitur]
        hasil_prediksi = knn.predict(fitur_final)
        
        # Voting
        jumlah_stres = np.sum(hasil_prediksi == 1)
        persen_stres = (jumlah_stres / len(hasil_prediksi)) * 100
        status_akhir = "STRES" if persen_stres > 50 else "RILEKS"
        
        # Cleanup
        if os.path.exists(file_path): os.remove(file_path)
        del raw, epochs, data_signal
        gc.collect()
        
        return jsonify({
            'filename': file.filename,
            'result': status_akhir,
            'confidence': round(persen_stres, 2) if status_akhir == "STRES" else round(100-persen_stres, 2),
            'channel_used': TARGET_CHANNELS
        })

    except ValueError as ve:
        # Error Validasi (Channel kurang, file rusak)
        if os.path.exists(file_path): os.remove(file_path)
        return jsonify({'error': str(ve)}), 400
        
    except Exception as e:
        # Error Server Lainnya
        error_msg = traceback.format_exc()
        print("❌ CRITICAL ERROR:")
        print(error_msg)
        if os.path.exists(file_path): os.remove(file_path)
        return jsonify({'error': "Terjadi kesalahan internal pada server."}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
