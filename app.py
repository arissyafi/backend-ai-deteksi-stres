import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import mne
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Load Model (Nanti file .pkl harus diupload juga)
# Render tidak punya Google Drive, jadi file model harus ada di folder yang sama
try:
    model_data = joblib.load('OTAK_AI_100FILE_FINAL.pkl')
    knn = model_data['model_knn']
    scaler = model_data['scaler']
    idx_fitur = model_data['fitur_idx']
    print("✅ Model AI Berhasil Dimuat!")
except:
    print("⚠️ Warning: Model tidak ditemukan. Pastikan file .pkl sudah diupload.")

# --- FUNGSI EKSTRAKSI (Sama seperti sebelumnya) ---
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
    return "Server AI NeuroSense Aktif 24/7!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    filename = file.filename
    file.save(filename)
    
    try:
        # Proses AI
        raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
        raw.filter(1.0, 45.0, fir_design='firwin', verbose=False)
        if raw.info['sfreq'] != 128: raw.resample(128, verbose=False)
        
        tgt_chs = ['AF3', 'T7', 'Pz', 'T8', 'AF4']
        current_chs = raw.ch_names
        picked = [ch for ch in tgt_chs if ch in current_chs]
        raw.pick_channels(picked)
        
        epochs = mne.make_fixed_length_epochs(raw, duration=2.0, overlap=1.0, verbose=False)
        data_signal = epochs.get_data()
        
        if len(data_signal) == 0:
            os.remove(filename)
            return jsonify({'error': 'File kosong/pendek'}), 400

        fitur_mentah = []
        for i in range(len(data_signal)):
            feat = extract_features_complete(data_signal[i])
            fitur_mentah.append(feat)
        
        fitur_mentah = np.array(fitur_mentah)
        fitur_terpilih = fitur_mentah[:, idx_fitur] 
        fitur_final = scaler.transform(fitur_terpilih)
        
        hasil_prediksi = knn.predict(fitur_final)
        
        jumlah_stres = np.sum(hasil_prediksi == 1)
        persen_stres = (jumlah_stres / len(hasil_prediksi)) * 100
        status_akhir = "STRES" if persen_stres > 50 else "RILEKS"
        
        os.remove(filename)
        
        return jsonify({
            'filename': filename,
            'result': status_akhir,
            'confidence': round(persen_stres, 2) if status_akhir == "STRES" else round(100-persen_stres, 2)
        })
        
    except Exception as e:
        if os.path.exists(filename): os.remove(filename)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Gunakan port dari environment variable Render atau default 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)