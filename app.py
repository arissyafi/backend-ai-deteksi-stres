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
CORS(app) # Penting agar Vercel bisa akses

# --- KONFIGURASI MODEL ---
MODEL_FILE = 'OTAK_AI_100FILE_FINAL.pkl'
model_data = None

# Load model di awal (Global)
if os.path.exists(MODEL_FILE):
    try:
        model_data = joblib.load(MODEL_FILE)
        print("✅ Model berhasil dimuat saat startup.")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
else:
    print(f"⚠️ Peringatan: File {MODEL_FILE} tidak ditemukan di server.")

# --- FUNGSI EKSTRAKSI (SAMA PERSIS) ---
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
    return "NeuroSense API is Running! (v2.0 Fix)"

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Cek Model
    if model_data is None:
        return jsonify({'error': 'Model AI tidak ditemukan di server'}), 500

    # 2. Cek File
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dikirim'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    # 3. Simpan File ke Folder Sementara (/tmp)
    # PENTING: Di Render/Cloud, kita tidak boleh tulis di root directory sembarangan
    try:
        temp_dir = '/tmp' 
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        print(f"📁 File disimpan sementara di: {file_path}")

    except Exception as e:
        print(f"❌ Error Saving File: {e}")
        return jsonify({'error': f"Gagal menyimpan file: {str(e)}"}), 500

    # 4. Proses AI (Dalam Try-Except agar error terlihat jelas)
    try:
        knn = model_data['model_knn']
        scaler = model_data['scaler']
        idx_fitur = model_data['fitur_idx']

        # Baca EDF
        # preload=True bisa memakan RAM, tapi untuk file kecil (beberapa MB) di Render biasanya masih aman.
        # Jika file >10MB, pertimbangkan preload=False atau potong durasi.
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Preprocessing
        raw.filter(1.0, 45.0, fir_design='firwin', verbose=False)
        if raw.info['sfreq'] != 128: raw.resample(128, verbose=False)
        
        tgt_chs = ['AF3', 'T7', 'Pz', 'T8', 'AF4']
        current_chs = raw.ch_names
        picked = [ch for ch in tgt_chs if ch in current_chs]
        
        if len(picked) < 3:
            raise ValueError(f"Channel tidak cukup. Ditemukan: {picked}")
            
        raw.pick_channels(picked)
        
        # Epoching
        epochs = mne.make_fixed_length_epochs(raw, duration=2.0, overlap=1.0, verbose=False)
        data_signal = epochs.get_data()
        
        if len(data_signal) == 0:
            raise ValueError("File valid tapi data sinyal kosong.")

        # Ekstraksi
        fitur_mentah = []
        for i in range(len(data_signal)):
            feat = extract_features_complete(data_signal[i])
            fitur_mentah.append(feat)
        
        fitur_mentah = np.array(fitur_mentah)
        
        # Seleksi & Prediksi
        fitur_terpilih = fitur_mentah[:, idx_fitur] 
        fitur_final = scaler.transform(fitur_terpilih)
        hasil_prediksi = knn.predict(fitur_final)
        
        # Voting
        jumlah_stres = np.sum(hasil_prediksi == 1)
        persen_stres = (jumlah_stres / len(hasil_prediksi)) * 100
        status_akhir = "STRES" if persen_stres > 50 else "RILEKS"
        
        # Bersihkan File & Memori
        if os.path.exists(file_path): os.remove(file_path)
        del raw, epochs, data_signal
        gc.collect()
        
        print(f"✅ Sukses Prediksi: {status_akhir} ({persen_stres:.2f}%)")
        
        return jsonify({
            'filename': file.filename,
            'result': status_akhir,
            'confidence': round(persen_stres, 2) if status_akhir == "STRES" else round(100-persen_stres, 2)
        })

    except Exception as e:
        # PENTING: Ini akan mencetak error detail ke Logs Render
        error_msg = traceback.format_exc()
        print("❌ CRITICAL ERROR DI PROSES AI:")
        print(error_msg)
        
        # Hapus file jika masih ada
        if os.path.exists(file_path): os.remove(file_path)
        
        return jsonify({'error': f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
