# ===============================================
# APP: Flight Price Explorer
# ===============================================
import os
import math
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import joblib
import streamlit as st
import base64

st.set_page_config(page_title="Flight Price Explorer (JFK ⇄ MIA)", layout="wide")

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
MODEL_PATH = BASE_DIR / "models" / "random_forest_flights_v2.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Prioridad: secrets → variable de entorno
DRIVE_ID = st.secrets.get("DRIVE_ID") or os.getenv("DRIVE_ID")
if not DRIVE_ID:
    st.error("❌ Falta configurar DRIVE_ID en st.secrets o variables de entorno.")
    st.stop()

DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"

# =======================
# DATOS DE AEROPUERTOS
# =======================
ORIGINS = ["JFK", "MIA"]
DESTS = [
    "ATL", "BOS", "CLT", "DEN", "DFW", "DTW", "IAD", "LAX", "MIA",
    "OAK", "ORD", "PHL", "SFO", "EWR", "JFK", "LGA"
]

AIRPORT_NAMES = {
    "ATL": "ATL (Atlanta, Georgia)", "BOS": "BOS (Boston, Massachusetts)", "CLT": "CLT (Charlotte, North Carolina)",
    "DEN": "DEN (Denver, Colorado)", "DFW": "DFW (Dallas/Fort Worth, Texas)", "DTW": "DTW (Detroit, Michigan)",
    "IAD": "IAD (Washington D.C.)", "LAX": "LAX (Los Ángeles, California)", "MIA": "MIA (Miami, Florida)",
    "OAK": "OAK (Oakland, California)", "ORD": "ORD (Chicago, Illinois)", "PHL": "PHL (Filadelfia, Pennsylvania)",
    "SFO": "SFO (San Francisco, California)", "EWR": "EWR (Newark, New Jersey)", "JFK": "JFK (Nueva York, NY)",
    "LGA": "LGA (Nueva York, NY)"
}

AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277), "BOS": (42.3656, -71.0096), "CLT": (35.2140, -80.9431),
    "DEN": (39.8561, -104.6737), "DFW": (32.8998, -97.0403), "DTW": (42.2124, -83.3534),
    "IAD": (38.9531, -77.4565), "LAX": (33.9416, -118.4085), "MIA": (25.7959, -80.2871),
    "OAK": (37.7126, -122.2197), "ORD": (41.9742, -87.9073), "PHL": (39.8744, -75.2424),
    "SFO": (37.6213, -122.3790), "EWR": (40.6895, -74.1745), "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740)
}

# =======================
# FUNCIONES AUXILIARES
# =======================
@st.cache_resource
def load_model():
    # Descarga siempre (por si actualizaste el modelo en Drive)
    try:
        with st.spinner("📥 Descargando modelo desde Google Drive..."):
            try:
                import gdown
            except ImportError:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
                import gdown

            # Descarga a archivo temporal y luego mueve (para evitar .pkl corrupto si la descarga se corta)
            tmp_path = MODEL_PATH.with_suffix(".tmp.pkl")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

            gdown.download(DRIVE_URL, str(tmp_path), quiet=True)

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                st.error("❌ La descarga del modelo falló o vino vacía. Verificá el DRIVE_ID/permiso de enlace.")
                st.stop()

            tmp_path.replace(MODEL_PATH)

        with st.spinner("🧠 Cargando modelo..."):
            model = joblib.load(MODEL_PATH)

        return model

    except Exception as e:
        st.error(f"❌ Error al descargar/cargar el modelo: {e}")
        st.stop()


def haversine_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = map(math.radians, [lat1, lat2])
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def route_distance_km(origin, dest):
    if origin not in AIRPORT_COORDS or dest not in AIRPORT_COORDS:
        return np.nan
    lat1, lon1 = AIRPORT_COORDS[origin]
    lat2, lon2 = AIRPORT_COORDS[dest]
    return haversine_distance_km(lat1, lon1, lat2, lon2)


def estimate_duration_min(distance_km, nonstop=True):
    if np.isnan(distance_km):
        return 120
    hours = distance_km / 800.0
    base_min = hours * 60 + 40
    return int(round(base_min + (0 if nonstop else 60)))


def infer_features_from_model(model):
    """Extrae columnas y categorías desde el modelo entrenado."""
    ct = model.named_steps["preprocess"]
    tmap = {name: (trans, cols) for name, trans, cols in ct.transformers_}
    cat_trans, cat_cols = tmap["cat"]
    num_trans, num_cols = tmap["num"]

    # Compatibilidad: cat_trans puede ser un Pipeline o un OneHotEncoder directo
    if hasattr(cat_trans, "named_steps"):
        oh = cat_trans.named_steps["onehot"]
    else:
        oh = cat_trans

    airlines = list(oh.categories_[cat_cols.index("main_airline")])
    dow_categories = list(oh.categories_[cat_cols.index("flight_dayofweek")])
    return list(num_cols), list(cat_cols), airlines, dow_categories


def build_features(origin, dest, flight_date, airline_code, is_refundable, nonstop,
                   num_cols, cat_cols, dow_categories):
    today = dt.date.today()
    days_to_departure = max((flight_date - today).days, 0)
    distance = route_distance_km(origin, dest)
    duration = estimate_duration_min(distance, nonstop)
    flight_month = flight_date.month
    flight_dayofweek = flight_date.strftime("%a") if isinstance(dow_categories[0], str) else flight_date.weekday()
    is_weekend = int(flight_date.weekday() in (5, 6))
    is_holiday_season = int(flight_month in (6, 7))

    row = {
        "days_to_departure": days_to_departure, "totalTravelDistance": distance,
        "duration_min": duration, "startingAirport": origin, "destinationAirport": dest,
        "isRefundable": int(is_refundable), "isNonStop": int(nonstop),
        "main_airline": airline_code, "flight_month": flight_month,
        "flight_dayofweek": flight_dayofweek, "flight_day": flight_date.day,
        "is_weekend": is_weekend, "is_holiday_season": is_holiday_season,
    }
    return pd.DataFrame([row])[list(num_cols) + list(cat_cols)]


def mostrar_tarjetas(df, origen, destino, titulo):
    """Muestra tarjetas visuales con el precio por aerolínea."""
    st.markdown(f"### {titulo}")
    for _, row in df.iterrows():
        nombre, precio = row["Aerolínea"], row["Precio"]
        clean_name = nombre.replace(" ", "_").replace(".", "").replace("-", "_")
        logo_html = "✈️"
        for name in [f"{clean_name}.png", f"{clean_name.lower()}.png", f"{clean_name.capitalize()}.png"]:
            path = BASE_DIR / "assets" / "logos" / name
            if path.exists():
                with open(path, "rb") as img:
                    logo_b64 = base64.b64encode(img.read()).decode()
                    logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="70">'
                    break

        st.markdown(f"""
        <div style='display:flex;align-items:center;justify-content:space-between;
        border:1px solid #ddd;border-radius:12px;padding:14px 20px;margin-bottom:10px;
        background:#fafafa;box-shadow:2px 2px 6px rgba(0,0,0,0.05);'>
            <div style='display:flex;align-items:center;gap:15px;'>
                {logo_html}
                <div><b>{nombre}</b><br><small>{origen} → {destino}</small></div>
            </div>
            <div><b style='color:#FF4B4B;font-size:1.3em;'>${precio:,.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)


# =======================
# CARGAR MODELO
# =======================
model = load_model()
num_cols, cat_cols, airlines_from_model, dow_categories = infer_features_from_model(model)

# =======================
# INTERFAZ VISUAL
# =======================
# --- Estilos CSS personalizados ---
st.markdown(
    """
    <style>
    /* ======= HEADER BANDERA ======= */
    .header-container {
        background: linear-gradient(180deg, #0A3161 70%, #B31942 70%);
        color: white;
        text-align: center;
        padding: 25px 10px 35px 10px;
        border-radius: 10px;
        position: relative;
        box-shadow: 0 3px 6px rgba(0,0,0,0.3);
    }
    .header-title {
        font-size: 2.2em;
        font-weight: 700;
        letter-spacing: 1px;
        color: white;
    }
    .header-stars {
        position: absolute;
        top: 8px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 18px;
        letter-spacing: 6px;
    }
    .subheader {
        font-size: 1em;
        color: #f1f1f1;
        font-style: italic;
    }

    /* ======= INPUTS Y BOTONES ======= */
    div[data-baseweb="select"] {
        border-radius: 6px;
    }
    div.stDateInput > div > input {
        border-radius: 6px !important;
        padding: 5px 10px !important;
    }
    div.stButton > button {
        background-color: #B31942 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        padding: 10px 0 !important;
    }
    div.stButton > button:hover {
        background-color: #861B2D !important;
        color: #fff !important;
        transform: scale(1.02);
    }

    /* ======= SEPARADORES ======= */
    hr {
        border: 1px solid #B31942;
        margin: 25px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Encabezado principal ---
st.markdown(
    """
    <div class="header-container">
        <div class="header-stars">★ ★ ★ ★ ★ ★</div>
        <div class="header-title">🇺🇸 Vuelos de Cabotaje EE. UU.</div>
        <div class="subheader">Explorá precios, aerolíneas y tendencias de vuelos domésticos</div>
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)

# --- Panel de búsqueda ---
trip_type = st.radio("✈️ Tipo de viaje", ["Ida y vuelta", "Solo ida"], horizontal=True)
c1, c2, c3, c4 = st.columns([1.3, 2, 1.3, 1.3])

with c1:
    origin_label = st.selectbox("🛫 Origen", [AIRPORT_NAMES[o] for o in ORIGINS])
    origin = [k for k, v in AIRPORT_NAMES.items() if v == origin_label][0]

with c2:
    dest_label = st.selectbox("🛬 Destino", [AIRPORT_NAMES[d] for d in DESTS if d != origin])
    dest = [k for k, v in AIRPORT_NAMES.items() if v == dest_label][0]

with c3:
    dep_date = st.date_input(
        "📅 Fecha de salida",
        value=dt.date.today() + dt.timedelta(days=14),
        min_value=dt.date.today(),
        format="DD/MM/YYYY"
    )

with c4:
    if trip_type == "Ida y vuelta":
        ret_date = st.date_input(
            "🔁 Fecha de vuelta",
            value=dt.date.today() + dt.timedelta(days=21),
            min_value=dep_date,
            format="DD/MM/YYYY"
        )
    else:
        ret_date = None
        st.write("")

st.markdown("<br>", unsafe_allow_html=True)
cta = st.button("🔍 Buscar vuelos", type="primary", use_container_width=True)
st.markdown("<hr>", unsafe_allow_html=True)

# =======================
# PREDICCIÓN
# =======================
def ejecutar_prediccion():
    X_out = pd.concat([
        build_features(origin, dest, dep_date, a, False, True, num_cols, cat_cols, dow_categories).assign(Aerolínea=a)
        for a in airlines_from_model
    ], ignore_index=True)
    y_out = model.predict(X_out)
    df_out = pd.DataFrame({"Aerolínea": X_out["Aerolínea"], "Precio": y_out}).sort_values("Precio")

    if trip_type == "Ida y vuelta" and ret_date:
        X_back = pd.concat([
            build_features(dest, origin, ret_date, a, False, True, num_cols, cat_cols, dow_categories).assign(Aerolínea=a)
            for a in airlines_from_model
        ], ignore_index=True)
        y_back = model.predict(X_back)
        df_back = pd.DataFrame({"Aerolínea": X_back["Aerolínea"], "Precio": y_back}).sort_values("Precio")
    else:
        df_back = None

    st.session_state.update({
        "df_air_out": df_out,
        "df_air_back": df_back,
        "origin": origin,
        "dest": dest,
        "trip_type": trip_type
    })


if cta:
    ejecutar_prediccion()

# =======================
# RESULTADOS Y GRÁFICOS
# =======================
if "df_air_out" in st.session_state:
    df_air_out = st.session_state.df_air_out
    df_air_back = st.session_state.df_air_back
    origin, dest, trip_type = st.session_state.origin, st.session_state.dest, st.session_state.trip_type

    st.markdown("## ✈️ Resultados por aerolínea")
    # ==============================
    # FILTRO DE AEROLÍNEAS (funcional y reactivo al primer clic)
    # ==============================
    todas_aerolineas = sorted(df_air_out["Aerolínea"].unique())
    
    # Inicializar estado solo una vez
    if "aerolineas_seleccionadas" not in st.session_state:
        st.session_state.aerolineas_seleccionadas = todas_aerolineas.copy()
    if "todas_seleccionadas" not in st.session_state:
        st.session_state.todas_seleccionadas = True
    
    with st.expander("🎯 Filtrar por Aerolínea", expanded=False):
        toggle_todas = st.checkbox(
            "Seleccionar todas las aerolíneas",
            value=st.session_state.todas_seleccionadas,
            help="Marcá o desmarcá para seleccionar o quitar todas las aerolíneas."
        )
    
        # --- Si cambia el toggle, actualiza y recarga ---
        if toggle_todas != st.session_state.todas_seleccionadas:
            st.session_state.todas_seleccionadas = toggle_todas
            st.session_state.aerolineas_seleccionadas = (
                todas_aerolineas.copy() if toggle_todas else []
            )
            st.rerun()
    
        # --- Multiselect para selección individual ---
        seleccion = st.multiselect(
            "Seleccioná las aerolíneas que quieras ver:",
            options=todas_aerolineas,
            default=st.session_state.aerolineas_seleccionadas,
            label_visibility="collapsed"
        )
    
        # --- Si cambia la selección manual, actualizar y recargar ---
        if set(seleccion) != set(st.session_state.aerolineas_seleccionadas):
            st.session_state.aerolineas_seleccionadas = seleccion
            st.session_state.todas_seleccionadas = len(seleccion) == len(todas_aerolineas)
            st.rerun()
    
        st.caption(
            f"🟩 Mostrando {len(st.session_state.aerolineas_seleccionadas)} "
            f"de {len(todas_aerolineas)} aerolíneas disponibles."
        )
    
    # Aplicar filtro directamente al DataFrame
    df_air_out_filtrado = df_air_out[
        df_air_out["Aerolínea"].isin(st.session_state.aerolineas_seleccionadas)
    ]

    # Análisis de precios
    st.markdown("### 📊 Análisis de precios")
    c1, c2, c3 = st.columns(3)
    with c1: tend = st.button("📈 Tendencia", use_container_width=True)
    with c2: aero = st.button("💵 Aerolíneas", use_container_width=True)
    with c3: temp = st.button("📅 Estacionalidad", use_container_width=True)

    for key in ["mostrar_tend", "mostrar_aero", "mostrar_temp"]:
        if key not in st.session_state:
            st.session_state[key] = False

    if tend:
        st.session_state.mostrar_tend = True
    if aero:
        st.session_state.mostrar_aero = True
    if temp:
        st.session_state.mostrar_temp = True

    # === Gráficos con botón de cierre ===
    def mostrar_grafico_tendencia():
        st.markdown("## 📈 Evolución del precio estimado")
        st.caption("Muestra cómo varía el precio estimado según los días de anticipación del vuelo.")
        dias = list(range(120, -1, -10))
        precios = []
        for d in dias:
            try:
                fecha_simulada = dep_date - dt.timedelta(days=d)
                pred = model.predict(
                    build_features(origin, dest, fecha_simulada,
                                   "Delta", False, True, num_cols, cat_cols, dow_categories)
                )[0]
                precios.append(pred)
            except Exception:
                precios.append(None)
        df_tend = pd.DataFrame({"Días antes del vuelo": dias, "Precio estimado (USD)": precios})
        chart = (
            alt.Chart(df_tend)
            .mark_line(point=True, color="#1E88E5", interpolate="monotone")
            .encode(
                x=alt.X("Días antes del vuelo:Q", sort="descending", title="Días antes del vuelo"),
                y=alt.Y("Precio estimado (USD):Q", title="Precio estimado (USD)"),
                tooltip=["Días antes del vuelo", "Precio estimado (USD)"]
            )
            .properties(width=900, height=450)
        )
        st.altair_chart(chart, use_container_width=True)
        st.button(
            "❌ Cerrar gráfico",
            key="close_tend",
            use_container_width=True,
            on_click=lambda: st.session_state.update({"mostrar_tend": False})
        )
    
    
    def mostrar_grafico_aerolineas():
        st.markdown("## 💵 Comparación de precios por aerolínea")
        st.caption("Compara los precios estimados promedio entre las aerolíneas para la ruta seleccionada.")
        df_sorted = df_air_out_filtrado.sort_values("Precio", ascending=True)
        df_sorted["Precio redondeado"] = df_sorted["Precio"].apply(round)
        chart = (
            alt.Chart(df_sorted)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                y=alt.Y("Aerolínea:N", sort="-x", title="Aerolínea"),
                x=alt.X("Precio redondeado:Q", title="Precio estimado (USD)"),
                color=alt.Color("Aerolínea:N", legend=None, scale=alt.Scale(scheme="tableau10")),
                tooltip=["Aerolínea", "Precio redondeado"]
            )
            .properties(width=900, height=450)
        )
        st.altair_chart(chart, use_container_width=True)
        st.button(
            "❌ Cerrar gráfico",
            key="close_aero",
            use_container_width=True,
            on_click=lambda: st.session_state.update({"mostrar_aero": False})
        )
    
    
    def mostrar_grafico_estacionalidad():
        st.markdown("## 📅 Evolución del precio promedio por mes")
        st.caption("Muestra cómo varían los precios estimados de los vuelos a lo largo del año, permitiendo identificar temporadas altas o bajas.")
        meses = list(range(1, 13))
        nombres_meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        precios = []
        for m in meses:
            try:
                pred = model.predict(
                    build_features(origin, dest, dt.date(2025, m, 15),
                                   "Delta", False, True, num_cols, cat_cols, dow_categories)
                )[0]
                precios.append(pred)
            except Exception:
                precios.append(None)
        df_mes = pd.DataFrame({
            "Mes": pd.Categorical(nombres_meses, categories=nombres_meses, ordered=True),
            "Precio promedio (USD)": precios
        })
        chart = (
            alt.Chart(df_mes)
            .mark_line(interpolate="monotone", color="#2E7D32", strokeWidth=3)
            .encode(
                x=alt.X("Mes:N", sort=nombres_meses, title="Mes del año"),
                y=alt.Y("Precio promedio (USD):Q", title="Precio promedio (USD)"),
                tooltip=["Mes", "Precio promedio (USD)"]
            )
            .properties(width=900, height=450)
            +
            alt.Chart(df_mes)
            .mark_point(filled=True, size=80, color="#43A047")
            .encode(
                x=alt.X("Mes:N", sort=nombres_meses),
                y="Precio promedio (USD):Q",
                tooltip=["Mes", "Precio promedio (USD)"]
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.button(
            "❌ Cerrar gráfico",
            key="close_temp",
            use_container_width=True,
            on_click=lambda: st.session_state.update({"mostrar_temp": False})
        )
    
    
    # === Mostrar gráficos activos ===
    if st.session_state.mostrar_tend:
        mostrar_grafico_tendencia()
    if st.session_state.mostrar_aero:
        mostrar_grafico_aerolineas()
    if st.session_state.mostrar_temp:
        mostrar_grafico_estacionalidad()


    # TARJETAS DE RESULTADOS
    st.markdown("---")
    tabs = st.tabs(["✈️ Vuelos de ida"] + (["🔁 Vuelos de vuelta"] if df_air_back is not None else []))
    with tabs[0]:
        mostrar_tarjetas(df_air_out_filtrado, origin, dest, "Vuelos de ida")
    if df_air_back is not None:
        with tabs[1]:
            mostrar_tarjetas(df_air_back, dest, origin, "Vuelos de vuelta")
