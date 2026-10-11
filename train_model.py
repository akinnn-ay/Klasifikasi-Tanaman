"""Membuat artefak model sesuai alur training notebook SDGs_ML_TANAMAN_V4."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "DATASET TANAMAN.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "tanaman_model.joblib"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan dalam dataset: {missing}")
    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Dataset memiliki nilai kosong pada fitur/label yang digunakan.")

    X = df[FEATURES].copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET].astype(str))

    # Menyamakan pembagian data dengan notebook: test_size=0.2, random_state=42,
    # stratify=y. Scaler hanya di-fit pada data latih.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Notebook memilih Random Forest saat akurasi seri karena urutan models
    # menempatkan Random Forest lebih dahulu dan hasil df_results stable-sort.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": FEATURES,
        "class_names": label_encoder.classes_.tolist(),
        "training_ranges": {
            col: {"min": float(X[col].min()), "max": float(X[col].max())}
            for col in FEATURES
        },
        "training_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "model_name": "Random Forest",
        "model_params": {"n_estimators": 100, "random_state": 42},
    }
    joblib.dump(artifact, MODEL_PATH)

    # Verifikasi sampel yang dipakai pada notebook.
    sample = pd.DataFrame([{
        "N": 90, "P": 42, "K": 43, "temperature": 21.0,
        "humidity": 82.0, "ph": 6.5, "rainfall": 203.0,
    }], columns=FEATURES)
    pred_code = int(model.predict(scaler.transform(sample))[0])
    pred_label = label_encoder.inverse_transform([pred_code])[0]
    probabilities = model.predict_proba(scaler.transform(sample))[0]
    print(f"Artefak tersimpan: {MODEL_PATH}")
    print(f"Model: Random Forest; fitur: {len(FEATURES)}")
    print(f"Kelas: {', '.join(label_encoder.classes_)}")
    print(f"Sampel notebook diprediksi sebagai: {pred_label}")
    print("Probabilitas:")
    for name, prob in sorted(zip(label_encoder.classes_, probabilities), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {prob:.2%}")


if __name__ == "__main__":
    main()
