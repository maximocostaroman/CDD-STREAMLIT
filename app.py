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
# CARGAR DATASET DE ENTRENAMIENTO
# =======================
@st.cache_data
def load_training_data():
    """Descarga el CSV original desde Google Drive y lo carga en memoria."""
    import gdown

    drive_id_data = st.secrets.get("DRIVE_ID_DATA") or os.getenv("DRIVE_ID_DATA")
    if not drive_id_data:
        st.error("❌ Falta configurar DRIVE_ID_DATA en secrets o variables de entorno.")
        st.stop()

    drive_url_data = f"https://drive.google.com/uc?id={drive_id_data}"
    dataset_path = BASE_DIR / "data" / "processed" / "flights_model_JFK_MIA.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dataset_path.with_suffix(".tmp.csv")
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)

    with st.spinner("📥 Descargando dataset CSV desde Google Drive..."):
        gdown.download(drive_url_data, str(tmp_path), quiet=False)

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        st.error("❌ La descarga del dataset falló o el archivo vino vacío. Verificá los permisos de Drive.")
        st.stop()

    tmp_path.replace(dataset_path)

    # Cargar CSV
    df = pd.read_csv(dataset_path)

    # Fechas y columnas derivadas
    if "flightDate" in df.columns:
        df["flightDate"] = pd.to_datetime(df["flightDate"], errors="coerce")
        df["flight_month"] = df["flightDate"].dt.month
        df["flight_month_name"] = df["flightDate"].dt.strftime("%b")
    else:
        st.warning("⚠️ El CSV no tiene la columna 'flightDate'. Algunos gráficos podrían no mostrarse.")

    return df

# =======================
# INTERFAZ VISUAL
st.markdown(
    """
    <style>
    /* === Encabezado bandera elegante === */
    .header-container {
        background: linear-gradient(180deg, #0A3161 65%, #B31942 65%);
        border-radius: 10px;
        padding: 35px 20px 45px 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        margin-bottom: 35px;
    }
    .header-stars {
        color: #ffffff;
        font-size: 16px;
        letter-spacing: 8px;
        margin-bottom: 6px;
        display: block;
    }
    .header-title {
        font-size: 2.1em;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .header-sub {
        font-size: 1em;
        color: #f1f1f1;
        font-style: italic;
    }

    /* === Inputs y botones === */
    div[data-baseweb="select"], div.stDateInput {
        border-radius: 8px !important;
        padding: 4px;
    }
    div.stDateInput > div > input {
        border-radius: 8px !important;
        padding: 8px 10px !important;
    }
    div.stButton > button {
        background-color: #B31942 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1.05em !important;
        padding: 10px 0 !important;
        transition: all 0.25s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #8F142F !important;
        transform: scale(1.02);
    }

    /* === Radio buttons === */
    div[role="radiogroup"] label {
        font-weight: 600;
    }

    /* === Tabs de Vuelos de ida / vuelta === */
    div[data-baseweb="tab-list"] {
        justify-content: center !important;
        margin-top: 15px;
        margin-bottom: 10px;
        gap: 10px;
    }

    button[data-baseweb="tab"] {
        background-color: #f9f9f9 !important;
        color: #333333 !important;
        font-weight: 700 !important;
        font-size: 1.1em !important;
        text-transform: uppercase !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 12px 24px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease-in-out;
        border-bottom: 4px solid transparent !important;
    }

    button[data-baseweb="tab"]:hover {
        background-color: #f1f1f1 !important;
        transform: scale(1.02);
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #B31942 !important;
        border-bottom: 4px solid #B31942 !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08) !important;
    }

    /* === Fondo general === */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f9f9f9 0%, #ffffff 60%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Encabezado bandera ---
st.markdown("""
<div class="header-container">
    <span class="header-stars">★ ★ ★ ★ ★ ★</span>
    <h1 class="header-title">Vuelos de Cabotaje EE. UU.</h1>
    <p class="header-sub">Explorá precios, aerolíneas y tendencias de vuelos domésticos</p>
</div>
""", unsafe_allow_html=True)

# ===============================================
# SECCIONES PRINCIPALES
# ===============================================
main_tab1, main_tab2 = st.tabs([
    "🧠 Predecí con nuestro modelo",
    "📊 Explorá nuestros datos"
])

# ====================================================
# SECCIÓN 1: PREDECÍ CON NUESTRO MODELO
# ====================================================
with main_tab1:
    # --- Panel de búsqueda ---
    st.markdown("### 🔍 Buscador de vuelos\n")
    st.markdown("✈️ Tipo de viaje: Solo ida")

    c1, c2, c3 = st.columns([1.2, 2, 1.5])

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

    st.markdown("<br>", unsafe_allow_html=True)
    cta = st.button("🔎 Buscar vuelos", type="primary", use_container_width=True)
    st.markdown("<br><hr>", unsafe_allow_html=True)

    # =======================
    # PREDICCIÓN
    # =======================
    def ejecutar_prediccion():
        X_out = pd.concat([
            build_features(origin, dest, dep_date, a, False, True,
                           num_cols, cat_cols, dow_categories).assign(Aerolínea=a)
            for a in airlines_from_model
        ], ignore_index=True)

        y_out = model.predict(X_out)
        df_out = pd.DataFrame({
            "Aerolínea": X_out["Aerolínea"],
            "Precio": y_out
        }).sort_values("Precio")

        st.session_state.update({
            "df_air_out": df_out,
            "origin": origin,
            "dest": dest
        })

    if cta:
        ejecutar_prediccion()

    # =======================
    # RESULTADOS Y GRÁFICOS
    # =======================
    if "df_air_out" in st.session_state:
        df_air_out = st.session_state.df_air_out
        origin, dest = st.session_state.origin, st.session_state.dest

        st.markdown("## ✈️ Resultados por aerolínea")

        # ==============================
        # FILTRO DE AEROLÍNEAS
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

            if toggle_todas != st.session_state.todas_seleccionadas:
                st.session_state.todas_seleccionadas = toggle_todas
                st.session_state.aerolineas_seleccionadas = (
                    todas_aerolineas.copy() if toggle_todas else []
                )
                st.rerun()

            seleccion = st.multiselect(
                "Seleccioná las aerolíneas que quieras ver:",
                options=todas_aerolineas,
                default=st.session_state.aerolineas_seleccionadas,
                label_visibility="collapsed"
            )

            if set(seleccion) != set(st.session_state.aerolineas_seleccionadas):
                st.session_state.aerolineas_seleccionadas = seleccion
                st.session_state.todas_seleccionadas = len(seleccion) == len(todas_aerolineas)
                st.rerun()

            st.caption(
                f"🟩 Mostrando {len(st.session_state.aerolineas_seleccionadas)} "
                f"de {len(todas_aerolineas)} aerolíneas disponibles."
            )

        # Aplicar filtro
        df_air_out_filtrado = df_air_out[
            df_air_out["Aerolínea"].isin(st.session_state.aerolineas_seleccionadas)
        ]

        # === Botones de gráficos ===
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

        # === Funciones de gráficos ===
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

        # === Tarjetas de resultados ===
        st.markdown("---")
        mostrar_tarjetas(df_air_out_filtrado, origin, dest, "Vuelos disponibles")

# ====================================================
# SECCIÓN 2: EXPLORÁ NUESTROS DATOS
# ====================================================
with main_tab2:
    st.markdown("## 📊 Explorá nuestros datos")
    st.caption("Los datos corresponden a vuelos reales utilizados para entrenar el modelo (abril–octubre 2022).")

    # === Cargar dataset real desde Drive ===
    df_data = load_training_data()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Precio por mes",
        "🏁 JFK vs MIA",
        "✈️ Aerolíneas",
        "⏰ Anticipación de compra",
        "🗺️ Mapa de rutas"
    ])

    with tab1:
        st.markdown("### 📅 Variación de precios promedio por mes (Abr–Oct 2022)")
        st.caption("Este gráfico muestra cómo varían los precios promedio de los vuelos durante el período de abril a octubre de 2022. Podés filtrar por aeropuerto de origen y destino para identificar tendencias estacionales o diferencias de precio.")
    
        # === Selectores ===
        col1, col2 = st.columns(2)
        with col1:
            origen_sel = st.selectbox("✈️ Aeropuerto de origen", ["Todos"] + ORIGINS)
        with col2:
            destino_sel = st.selectbox("🏁 Aeropuerto de destino", ["Todos"] + sorted(df_data["destinationAirport"].unique()))
    
        # === Filtrado ===
        df_filt = df_data.copy()
        if origen_sel != "Todos":
            df_filt = df_filt[df_filt["startingAirport"] == origen_sel]
        if destino_sel != "Todos":
            df_filt = df_filt[df_filt["destinationAirport"] == destino_sel]
    
        # === Preprocesamiento de fechas ===
        df_filt["flightDate"] = pd.to_datetime(df_filt["flightDate"], errors="coerce")
        df_filt = df_filt[df_filt["flightDate"].notna()]
        df_filt["month"] = df_filt["flightDate"].dt.month
        df_filt["month_name"] = df_filt["flightDate"].dt.strftime("%b")
    
        # === Filtrar solo meses del dataset ===
        meses_validos = [4, 5, 6, 7, 8, 9, 10]  # Abril a Octubre
        df_filt = df_filt[df_filt["month"].isin(meses_validos)]
    
        # === Agrupar por mes ===
        df_mes = (
            df_filt.groupby("month_name", as_index=False)["totalFare"]
            .mean()
            .rename(columns={"totalFare": "Precio promedio (USD)"})
        )
    
        # === Ordenar meses cronológicamente ===
        orden_meses = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
        df_mes["month_name"] = pd.Categorical(df_mes["month_name"], categories=orden_meses, ordered=True)
        df_mes = df_mes.sort_values("month_name")
    
        # === Crear gráfico ===
        chart = (
            alt.Chart(df_mes)
            .mark_line(point=alt.OverlayMarkDef(color="#0A3161", size=80), color="#B31942", strokeWidth=3)
            .encode(
                x=alt.X("month_name:N", title="Mes del año (2022)", sort=orden_meses),
                y=alt.Y("Precio promedio (USD):Q", title="Precio promedio (USD)", scale=alt.Scale(zero=False)),
                tooltip=["month_name", "Precio promedio (USD)"]
            )
            .properties(width=850, height=420)
            .configure_axis(labelFontSize=13, titleFontSize=14)
            .configure_title(fontSize=18)
        )
    
        st.altair_chart(chart, use_container_width=True)
    
        # === Leyenda descriptiva ===
        st.markdown(
            """
            <p style='font-size: 0.95em; color: #555; margin-top: 8px;'>
            <b>Interpretación:</b> este gráfico permite observar la evolución mensual de los precios promedio de los vuelos en el periodo
            analizado (abril–octubre 2022). Los picos o descensos pueden indicar temporadas de alta o baja demanda según la ruta seleccionada.
            </p>
            """,
            unsafe_allow_html=True,
        )

        with tab2:
            st.markdown("### 🏁 Comparativo de precios: JFK vs MIA")
            st.caption("Analizá cómo varían los precios promedio de vuelos según el aeropuerto de origen (JFK o MIA) hacia un destino específico durante abril–octubre 2022.")
    
            destino_sel = st.selectbox("🏙️ Seleccioná un destino", sorted(df_data["destinationAirport"].unique()))
    
            # Filtrar y agrupar
            df_comp = df_data[df_data["destinationAirport"] == destino_sel].copy()
            df_comp["flightDate"] = pd.to_datetime(df_comp["flightDate"], errors="coerce")
            df_comp["month"] = df_comp["flightDate"].dt.month
            df_comp["month_name"] = df_comp["flightDate"].dt.strftime("%b")
            meses_validos = [4, 5, 6, 7, 8, 9, 10]
            df_comp = df_comp[df_comp["month"].isin(meses_validos)]
    
            df_comp = (
                df_comp.groupby(["startingAirport", "month_name"], as_index=False)["totalFare"]
                .mean()
                .rename(columns={"totalFare": "Precio promedio (USD)"})
            )
    
            orden_meses = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
            df_comp["month_name"] = pd.Categorical(df_comp["month_name"], categories=orden_meses, ordered=True)
            df_comp = df_comp.sort_values("month_name")
    
            chart = (
                alt.Chart(df_comp)
                .mark_line(point=True, strokeWidth=3)
                .encode(
                    x=alt.X("month_name:N", title="Mes del año (2022)", sort=orden_meses),
                    y=alt.Y("Precio promedio (USD):Q", title="Precio promedio (USD)", scale=alt.Scale(zero=False)),
                    color=alt.Color("startingAirport:N", title="Origen", scale=alt.Scale(domain=["JFK", "MIA"], range=["#0A3161", "#B31942"])),
                    tooltip=["startingAirport", "month_name", "Precio promedio (USD)"]
                )
                .properties(width=850, height=420)
            )
    
            st.altair_chart(chart, use_container_width=True)
            st.markdown(
                "<p style='font-size:0.95em;color:#555;'>Permite observar si los precios difieren según el aeropuerto de salida (JFK o MIA) para un mismo destino, destacando posibles ventajas estacionales.</p>",
                unsafe_allow_html=True
            )
    
        with tab3:
            st.markdown("### 💰 Ranking de aerolíneas por precio promedio (Abr–Oct 2022)")
            st.caption("Visualizá el precio promedio de los vuelos por aerolínea, ordenadas de mayor a menor costo.")
        
            df_rank = df_data.copy()
            df_rank["flightDate"] = pd.to_datetime(df_rank["flightDate"], errors="coerce")
            df_rank["month"] = df_rank["flightDate"].dt.month
            df_rank = df_rank[df_rank["month"].isin([4, 5, 6, 7, 8, 9, 10])]
        
            df_rank = (
                df_rank.groupby("main_airline", as_index=False)["totalFare"]
                .mean()
                .rename(columns={"totalFare": "Precio promedio (USD)"})
                .sort_values("Precio promedio (USD)", ascending=False)
            )
        
            chart = (
                alt.Chart(df_rank)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Precio promedio (USD):Q", title="Precio promedio (USD)"),
                    y=alt.Y("main_airline:N", sort="-x", title="Aerolínea"),
                    color=alt.Color("Precio promedio (USD):Q", scale=alt.Scale(scheme="reds"), legend=None),
                    tooltip=["main_airline", "Precio promedio (USD)"]
                )
                .properties(width=850, height=450)
            )
        
            st.altair_chart(chart, use_container_width=True)
            st.markdown(
                "<p style='font-size:0.95em;color:#555;'>Las aerolíneas ubicadas más arriba presentan tarifas promedio más elevadas en el periodo analizado, reflejando posibles diferencias de segmento o cobertura de rutas.</p>",
                unsafe_allow_html=True,
            )

    
        with tab4:
            st.markdown("### ⏰ Efecto de la anticipación en el precio (por destino)")
            st.caption("Explorá cómo varía el precio promedio según la cantidad de días de anticipación con la que se compra el vuelo para cada destino (abril–octubre 2022).")
        
            destino_sel = st.selectbox(
                "🏙️ Seleccioná un destino",
                sorted(df_data["destinationAirport"].unique()),
                key="destino_anticipacion"
            )
            df_ant = df_data[df_data["destinationAirport"] == destino_sel].copy()
            df_ant = df_ant[df_ant["days_to_departure"].between(0, 120)]
            df_ant["flightDate"] = pd.to_datetime(df_ant["flightDate"], errors="coerce")
            df_ant = df_ant[df_ant["flightDate"].dt.month.isin([4, 5, 6, 7, 8, 9, 10])]
        
            df_ant = (
                df_ant.groupby("days_to_departure", as_index=False)["totalFare"]
                .mean()
                .rename(columns={"totalFare": "Precio promedio (USD)"})
                .sort_values("days_to_departure", ascending=False)
            )
        
            chart = (
                alt.Chart(df_ant)
                .mark_line(point=True, color="#1E88E5", strokeWidth=3)
                .encode(
                    x=alt.X("days_to_departure:Q", title="Días de anticipación", sort="descending"),
                    y=alt.Y("Precio promedio (USD):Q", title="Precio promedio (USD)", scale=alt.Scale(zero=False)),
                    tooltip=["days_to_departure", "Precio promedio (USD)"]
                )
                .properties(width=850, height=420)
            )
        
            st.altair_chart(chart, use_container_width=True)
            st.markdown(
                "<p style='font-size:0.95em;color:#555;'>El gráfico permite analizar la relación entre el precio y los días de anticipación para el destino seleccionado, identificando posibles patrones de demanda o variaciones estacionales.</p>",
                unsafe_allow_html=True,
            )


    
        with tab5:
            import pydeck as pdk
    
            st.markdown("### 🗺️ Mapa interactivo de rutas")
            st.caption("Visualizá las principales rutas entre aeropuertos de EE.UU., coloreadas según el precio promedio observado (abril–octubre 2022).")
    
            df_map = (
                df_data.groupby(["startingAirport", "destinationAirport"], as_index=False)
                .agg({"totalFare": "mean"})
                .rename(columns={"totalFare": "Precio promedio (USD)"})
            )
    
            df_map = df_map.merge(
                pd.DataFrame(AIRPORT_COORDS).T.reset_index().rename(columns={"index": "code", 0: "lat", 1: "lon"}),
                left_on="startingAirport", right_on="code"
            ).rename(columns={"lat": "lat_start", "lon": "lon_start"})
    
            df_map = df_map.merge(
                pd.DataFrame(AIRPORT_COORDS).T.reset_index().rename(columns={"index": "code", 0: "lat", 1: "lon"}),
                left_on="destinationAirport", right_on="code"
            ).rename(columns={"lat": "lat_end", "lon": "lon_end"})
    
            layer = pdk.Layer(
                "ArcLayer",
                data=df_map,
                get_source_position=["lon_start", "lat_start"],
                get_target_position=["lon_end", "lat_end"],
                get_tilt=15,
                get_width=2,
                get_source_color=[10, 49, 97, 180],
                get_target_color=[179, 25, 66, 180],
                pickable=True,
                auto_highlight=True,
            )
    
            view_state = pdk.ViewState(latitude=37.5, longitude=-96, zoom=3.5, pitch=30)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{startingAirport} → {destinationAirport}\n${Precio promedio (USD)}"}))
    
            st.markdown(
                "<p style='font-size:0.95em;color:#555;'>El mapa permite identificar visualmente las rutas más activas y las de mayor costo promedio dentro del periodo analizado.</p>",
                unsafe_allow_html=True
            )
