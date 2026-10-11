from pathlib import Path
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

import joblib
print("Joblib location:", joblib.__file__)

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "tanaman_model.joblib"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

st.set_page_config(
    page_title="AI Rekomendasi Tanaman",
    page_icon="🌱",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-title {font-size:2.15rem;font-weight:800;line-height:1.15;margin-bottom:.25rem;}
    .subtitle {color:#64748b;margin-bottom:1.25rem;}
    div[data-testid="stMetric"] {background:#f0fdf4;border:1px solid #bbf7d0;padding:1rem;border-radius:14px;}
    </style>
    <div class="main-title">🌱 AI Rekomendasi Tanaman</div>
    <div class="subtitle">Uji model klasifikasi tanaman berdasarkan unsur hara tanah dan kondisi iklim.</div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Memuat model...")
def load_artifact(path_string: str, modified_time: float):
    # Artefak harus berasal dari proyek ini/sumber yang dipercaya.
    return joblib.load(path_string)

if not MODEL_PATH.exists():
    st.error("Artefak model belum ditemukan: model/tanaman_model.joblib")
    st.info("Jalankan `python train_model.py` dari folder proyek untuk membuat artefak dari dataset dan alur training notebook.")
    st.stop()

try:
    artifact = load_artifact(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    model = artifact["model"]
    scaler = artifact["scaler"]
    label_encoder = artifact["label_encoder"]
    feature_names = artifact["feature_names"]
    training_ranges = artifact.get("training_ranges", {})
except Exception as exc:
    st.error("Model gagal dimuat. Periksa file artefak dan versi dependensi.")
    st.caption(f"Detail teknis: {type(exc).__name__}")
    st.stop()

st.caption(f"Model: {artifact.get('model_name', 'Klasifikasi')} · {artifact.get('training_rows', '—')} baris dataset training · {len(feature_names)} fitur")

with st.form("prediction_form"):
    st.subheader("Masukkan kondisi lahan")
    st.caption("Nilai awal diisi sesuai contoh inferensi dalam notebook. Silakan ubah sesuai data yang ingin diuji.")
    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0, step=1.0, help="Nilai numerik kandungan nitrogen sesuai satuan dataset.")
        p = st.number_input("Fosfor (P)", min_value=0.0, value=42.0, step=1.0, help="Nilai numerik kandungan fosfor sesuai satuan dataset.")
        k = st.number_input("Kalium (K)", min_value=0.0, value=43.0, step=1.0, help="Nilai numerik kandungan kalium sesuai satuan dataset.")
        temperature = st.number_input("Suhu (°C)", min_value=-50.0, max_value=70.0, value=21.0, step=0.1)
    with c2:
        humidity = st.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
        ph = st.number_input("pH tanah", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        rainfall = st.number_input("Curah hujan (mm)", min_value=0.0, value=203.0, step=1.0)

    submitted = st.form_submit_button("🔍 Prediksi Tanaman", type="primary", use_container_width=True)

if submitted:
    values = {
        "N": n, "P": p, "K": k, "temperature": temperature,
        "humidity": humidity, "ph": ph, "rainfall": rainfall,
    }
    if not all(np.isfinite(float(v)) for v in values.values()):
        st.error("Semua input harus berupa angka yang valid.")
        st.stop()

    input_df = pd.DataFrame([[values[col] for col in feature_names]], columns=feature_names)
    out_of_range = []
    for col in feature_names:
        bounds = training_ranges.get(col)
        if bounds and (values[col] < bounds["min"] or values[col] > bounds["max"]):
            out_of_range.append(f"{col}: rentang dataset {bounds['min']:.2f}–{bounds['max']:.2f}")
    if out_of_range:
        st.warning("Sebagian input berada di luar rentang yang terlihat pada dataset training. Prediksi tetap dihitung, tetapi hasilnya mungkin kurang andal.\n\n" + "\n".join(f"- {x}" for x in out_of_range))

    try:
        scaled_input = scaler.transform(input_df)
        pred_code = int(model.predict(scaled_input)[0])
        predicted_label = str(label_encoder.inverse_transform([pred_code])[0])
        st.divider()
        st.subheader("Hasil prediksi")
        st.success(f"Tanaman yang diprediksi: **{predicted_label.replace('_', ' ').title()}**")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            # Kode keluaran predict_proba berkorespondensi dengan model.classes_.
            class_codes = [int(x) for x in model.classes_]
            rows = []
            for code, probability in zip(class_codes, probabilities):
                label = str(label_encoder.inverse_transform([code])[0])
                rows.append({"Tanaman": label.replace("_", " ").title(), "Probabilitas model": float(probability)})
            result_df = pd.DataFrame(rows).sort_values("Probabilitas model", ascending=False).reset_index(drop=True)
            top_probability = float(result_df.iloc[0]["Probabilitas model"])
            st.metric("Probabilitas kelas teratas", f"{top_probability:.1%}")
            st.caption("Probabilitas adalah keluaran model, bukan jaminan keberhasilan tumbuh atau panen.")
            st.markdown("**Urutan kelas berdasarkan probabilitas model**")
            chart_df = result_df.set_index("Tanaman")
            st.bar_chart(chart_df, x_label="Tanaman", y_label="Probabilitas", horizontal=True)
            display_df = result_df.copy()
            display_df["Probabilitas model"] = display_df["Probabilitas model"].map(lambda x: f"{x:.2%}")
            st.dataframe(display_df, hide_index=True, use_container_width=True)

        with st.expander("Lihat kembali input yang diuji"):
            st.dataframe(input_df, hide_index=True, use_container_width=True)
        st.caption("Hasil ini merupakan prediksi model berdasarkan dataset dan preprocessing pada proyek, bukan saran agronomi yang dijamin.")
    except Exception as exc:
        st.error("Prediksi gagal dijalankan. Periksa kesesuaian artefak model dan input.")
        st.caption(f"Detail teknis: {type(exc).__name__}")

with st.expander("Tentang model"):
    st.write("Aplikasi menggunakan Random Forest dan StandardScaler yang disimpan sebagai artefak model. Tujuh fitur dipakai mengikuti bagian inferensi notebook.")
    st.write("Kelas yang tersedia: " + ", ".join(str(x) for x in label_encoder.classes_) + ".")
    st.write("Metrik evaluasi yang tercatat pada notebook tidak menjamin hasil yang sama untuk data lapangan baru.")
