# AI Rekomendasi Tanaman — Streamlit

Aplikasi sederhana untuk menguji model klasifikasi tanaman berdasarkan tujuh fitur: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, dan `rainfall`.

## Audit notebook dan keputusan implementasi

Notebook `SDGs_ML_TANAMAN_V4 (2).ipynb` membandingkan Random Forest dan Gaussian Naive Bayes. Dalam output notebook, keduanya mendapat akurasi 1.00 pada pembagian train/test dan 5-fold CV. Karena `df_results` diurutkan berdasarkan akurasi dan Random Forest berada pertama saat seri, implementasi ini memakai **RandomForestClassifier(n_estimators=100, random_state=42)**.

Notebook melakukan `LabelEncoder` pada target, pembagian data `test_size=0.2, random_state=42, stratify=y`, `StandardScaler` yang di-fit pada data latih, dan melatih model final pada tujuh fitur. Tidak ada file model terpisah yang disertakan dalam unggahan, jadi `model/tanaman_model.joblib` dibuat dengan mereproduksi alur tersebut dari CSV. Ini merupakan rekonstruksi artefak berdasarkan kode notebook, bukan file model asli yang sebelumnya disimpan.

## Isi proyek

- `app.py` — aplikasi Streamlit.
- `train_model.py` — membuat artefak model dari dataset dengan alur yang mengikuti notebook.
- `model/tanaman_model.joblib` — artefak model, scaler, encoder, dan rentang fitur.
- `DATASET TANAMAN.csv` — dataset yang digunakan untuk membuat artefak.
- `requirements.txt` — dependensi.

## Menjalankan lokal

Disarankan menggunakan Python 3.11 atau 3.12.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Streamlit akan menampilkan alamat lokal, biasanya `http://localhost:8501`.

## Deployment online: Streamlit Community Cloud

1. Buat repository GitHub baru.
2. Unggah `app.py`, `requirements.txt`, `model/tanaman_model.joblib`, dan `README.md`. Dataset serta `train_model.py` boleh disertakan untuk dokumentasi/reproduksi, tetapi aplikasi yang sudah memiliki artefak tidak membutuhkannya saat runtime.
3. Pastikan `model/tanaman_model.joblib` masuk ke repository dan ukurannya sesuai batas GitHub/hosting. Jangan commit secret atau data pribadi.
4. Buka https://share.streamlit.io/ dan masuk menggunakan GitHub.
5. Pilih repository, branch, dan file utama `app.py`.
6. Deploy. Tunggu proses instalasi dependensi selesai.
7. Buka URL publik yang diberikan dan uji prediksi dengan nilai contoh.

Jika ingin membuat ulang artefak, jalankan `python train_model.py` secara lokal lalu commit file model yang dihasilkan. Jangan melatih ulang model saat aplikasi dimulai.

## Sampel notebook untuk pengujian

- N = 90
- P = 42
- K = 43
- temperature = 21.0
- humidity = 82.0
- ph = 6.5
- rainfall = 203.0

Output notebook yang terlihat menunjukkan sampel ini diprediksi sebagai `rice` dengan probabilitas model 100%. Setelah menjalankan `train_model.py`, bandingkan hasil aplikasi dengan output tersebut.

## Catatan penting

- Rentang data yang terlihat di dataset ditampilkan sebagai peringatan, bukan batas mutlak.
- Probabilitas Random Forest bukan jaminan keberhasilan tumbuh/panen.
- Artefak `joblib`/pickle hanya boleh dimuat jika berasal dari sumber tepercaya.
- Untuk konsistensi, deploy dengan versi dependensi pada `requirements.txt`.
