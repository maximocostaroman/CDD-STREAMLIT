import os
import gdown
import joblib
import streamlit as st

# --- Ruta local donde guardaremos el modelo ---
MODEL_PATH = "models/random_forest_flights_v2.pkl"

# --- URL de descarga desde Google Drive (reemplazá con tu ID real) ---
DRIVE_ID = "TU_ID_AQUI"  # 👈 reemplazá con el tuyo
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"

# --- Descargar modelo si no existe localmente ---
if not os.path.exists(MODEL_PATH):
    st.info("Descargando modelo desde Google Drive... Esto puede tardar varios minutos ⏳")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    st.success("✅ Modelo descargado correctamente.")

# --- Cargar modelo ---
st.info("Cargando modelo en memoria...")
modelo = joblib.load(MODEL_PATH)
st.success("Modelo cargado y listo para usar ✅")
