import json # Para leer metricas.json
import math # log1p/exp para el veredicto, floor para la edad
from pathlib import Path # Rutas relativas a ESTE archivo (funciona desde cualquier carpeta)

import pandas as pd # "pd" es el alias de pandas
import streamlit as st # "st" es el alias de streamlit

# Configuracion de la pagina (tiene que ir antes que cualquier otro st.)
st.set_page_config(page_title="Football Value Intelligence", page_icon="⚽")

CARPETA = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# La app NO entrena ni corre el modelo. Todo se pre-calcula con preparar_datos.py
# y queda en app_data.csv + metricas.json. Aca solo los leemos.
# ----------------------------------------------------------------------------

@st.cache_data
def cargar_datos():
    df = pd.read_csv(CARPETA / "app_data.csv")
    with open(CARPETA / "metricas.json", encoding="utf-8") as f:
        meta = json.load(f)
    return df, meta

df, meta = cargar_datos()

# ---- Traducciones para mostrar (los datos vienen en ingles) ----
POSICION = {"Attack": "Delantero", "Midfield": "Mediocampista", "Defender": "Defensor",
            "Goalkeeper": "Arquero", "Missing": "Sin dato"}
SUB_POSICION = {
    "Centre-Forward": "Delantero centro", "Second Striker": "Segundo delantero",
    "Left Winger": "Extremo izquierdo", "Right Winger": "Extremo derecho",
    "Attacking Midfield": "Mediapunta", "Central Midfield": "Mediocentro",
    "Defensive Midfield": "Mediocentro defensivo", "Left Midfield": "Volante izquierdo",
    "Right Midfield": "Volante derecho", "Centre-Back": "Defensor central",
    "Left-Back": "Lateral izquierdo", "Right-Back": "Lateral derecho",
    "Goalkeeper": "Arquero", "Missing": "Sin dato",
}
PIE = {"right": "Derecho", "left": "Izquierdo", "both": "Ambidiestro", "unknown": "Sin dato"}


# ---- Funciones de formato (todas toleran datos faltantes) ----
def formato_euros(valor):
    """1500000 -> €1.500.000 (punto como separador de miles, € adelante)."""
    return "€" + f"{valor:,.0f}".replace(",", ".")

def miles(n):
    """19044 -> '19.044'."""
    return f"{n:,.0f}".replace(",", ".")

def entero(valor, sin_dato="—"):
    """Numero redondeado con punto de miles: 8317.4 -> '8.317'."""
    return sin_dato if pd.isna(valor) else miles(round(valor))

def edad(valor):
    """La edad se TRUNCA, no se redondea: 18,9 años son 18 (todavia no cumplio 19)."""
    return "—" if pd.isna(valor) else str(math.floor(valor))

def fecha_linda(iso):
    """'2026-05-24' -> '24/05/2026'."""
    return "—" if pd.isna(iso) else "/".join(reversed(str(iso)[:10].split("-")))


# ---------------- INTERFAZ ----------------

st.title("⚽ Football Value Intelligence")
st.subheader("Detectá jugadores subvalorados y sobrevalorados")
st.caption(
    f"Datos de Transfermarkt al {fecha_linda(meta['fecha_ref'])} · "
    f"{miles(len(df))} jugadores activos · "
    "Predicciones evaluadas con validación cruzada"
)

# ---- Filtro por liga (etiqueta con pais: ya no se mezclan ligas homonimas) ----
ligas_disponibles = sorted(df["liga"].dropna().unique().tolist())
liga = st.selectbox("Filtrá por liga (opcional):", ["Todas las ligas"] + ligas_disponibles)
df_filtrado = df if liga == "Todas las ligas" else df[df["liga"] == liga]

# ---- Buscador ----
# Usamos el player_id como valor y mostramos "Nombre — Club (edad)".
# Antes eligiamos por NOMBRE, y con 440 nombres repetidos (ej: 3 "Adama Traoré")
# la app podia mostrar al jugador equivocado. Edad y posicion distinguen los
# casos con mismo nombre Y mismo club (ej: dos "Dudu" en Athletico Paranaense).
etiquetas = {
    pid: f"{nombre} — {club} ({edad(a)} años, {POSICION.get(pos, pos)})"
    for pid, nombre, club, a, pos in zip(
        df_filtrado["player_id"], df_filtrado["name"], df_filtrado["club_nombre"],
        df_filtrado["age"], df_filtrado["position"],
    )
}
ids_ordenados = sorted(etiquetas, key=lambda pid: etiquetas[pid])

player_id = st.selectbox(
    "Buscá un jugador:",
    options=ids_ordenados,
    index=None,
    format_func=lambda pid: etiquetas[pid],
    placeholder="Escribí un nombre...",
)

if player_id is not None: # Solo se ejecuta si el usuario selecciono algo
    j = df_filtrado[df_filtrado["player_id"] == player_id].iloc[0]
    cubierto = bool(j["liga_con_estadisticas"])
    m = meta["metricas"]

    # ---- Encabezado del jugador ----
    st.markdown(f"### {j['name']}")
    st.write(f"{j['club_nombre']} · {j['liga']}")

    # ---- Valor real vs modelo ----
    valor_predicho = j["valor_predicho"]
    col1, col2 = st.columns(2)

    if pd.isna(j["market_value_in_eur"]):
        col1.metric("Valor Transfermarkt", "Sin dato")
        col2.metric("Valor según el modelo", formato_euros(valor_predicho))
        st.info("Transfermarkt no tiene valor cargado para este jugador. "
                "Mostramos solo la estimación del modelo.")
    else:
        valor_real = j["market_value_in_eur"]
        col1.metric("Valor Transfermarkt", formato_euros(valor_real))
        col2.metric("Valor según el modelo", formato_euros(valor_predicho))

        # Comparamos en escala logaritmica (igual que el modelo) contra el
        # error tipico del modelo. Si la diferencia es menor, es ruido.
        ratio_log = math.log1p(valor_predicho) - math.log1p(valor_real) # misma escala que el modelo
        umbral = j["umbral_log"]
        margen_arriba = (math.exp(umbral) - 1) * 100
        margen_abajo = (1 - math.exp(-umbral)) * 100
        porcentaje = (valor_predicho / valor_real - 1) * 100

        if ratio_log > umbral:
            st.success(f"📈 **SUBVALORADO** — el modelo lo valúa {porcentaje:+.0f}% "
                       f"({formato_euros(valor_predicho - valor_real)} más). "
                       f"Supera el margen de error típico del modelo (+{margen_arriba:.0f}%).")
        elif ratio_log < -umbral:
            st.error(f"📉 **SOBREVALORADO** — el modelo lo valúa {porcentaje:+.0f}% "
                     f"({formato_euros(valor_real - valor_predicho)} menos). "
                     f"Supera el margen de error típico del modelo (−{margen_abajo:.0f}%).")
        else:
            st.info(f"✅ **VALOR JUSTO** — diferencia de {porcentaje:+.0f}%, dentro del margen "
                    f"de error típico del modelo (−{margen_abajo:.0f}% / +{margen_arriba:.0f}%).")

    # ---- Confiabilidad segun cobertura de datos ----
    if cubierto:
        r2 = m["modelo_ligas_con_estadisticas"]["r2_log"]
        st.caption(f"🟢 Confiabilidad alta: esta liga tiene estadísticas partido a partido (R² {r2:.2f}).")
    else:
        r2 = m["modelo_ligas_sin_estadisticas"]["r2_log"]
        st.caption(f"🟡 Confiabilidad media: sin estadísticas partido a partido para esta liga. "
                   f"La estimación usa perfil, club y experiencia previa (R² {r2:.2f}).")

    # ---- Ficha (disponible para TODOS los jugadores) ----
    st.divider()
    st.markdown("**Ficha**")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Edad", edad(j["age"]))
    f2.metric("Posición", SUB_POSICION.get(j["sub_position"], POSICION.get(j["position"], j["position"])))
    f3.metric("Pie", PIE.get(j["foot"], j["foot"]))
    f4.metric("Altura", "—" if pd.isna(j["height_in_cm"]) else f"{int(j['height_in_cm'])} cm")

    f5, f6, f7 = st.columns(3)
    f5.metric("Contrato hasta", fecha_linda(j["contract_expiration_date"]))
    f6.metric("Partidos con la selección", entero(j["international_caps"]))
    f7.metric("Goles con la selección", entero(j["international_goals"]))

    # ---- Rendimiento reciente (solo si hay datos reales) ----
    st.divider()
    st.markdown(f"**Últimos 2 años** ({fecha_linda(meta['inicio_ventana'])} al {fecha_linda(meta['fecha_ref'])})")
    if cubierto:
        c1, c2, c3 = st.columns(3)
        c1.metric("Partidos", entero(j["games_2y"]))
        c2.metric("Minutos", entero(j["minutes_2y"]))
        c3.metric("Min/partido", entero(j["minutes_per_game_2y"]))
        c4, c5, c6 = st.columns(3)
        c4.metric("Goles", entero(j["goals_2y"]))
        c5.metric("Asistencias", entero(j["assists_2y"]))
        c6.metric("Partidos en copas UEFA", entero(j["euro_games_2y"]))
    else:
        st.info(f"El dataset no tiene estadísticas partido a partido de su club actual "
                f"({j['liga']}). No mostramos ceros para no confundir \"no jugó\" con \"no hay datos\".")

    # ---- Experiencia previa (disponible para todos) ----
    st.markdown("**Carrera en las 14 ligas europeas cubiertas y copas UEFA**")
    e1, e2, e3 = st.columns(3)
    e1.metric("Partidos", entero(j["career_games"]))
    e2.metric("Goles", entero(j["career_goals"]))
    e3.metric("Asistencias", entero(j["career_assists"]))

    # ---- Contexto del club ----
    st.markdown("**Su club en la liga (últimos 2 años)**")
    k1, k2, k3 = st.columns(3)
    k1.metric("Posición en la tabla", "—" if pd.isna(j["club_position"]) else f"{int(j['club_position'])}°")
    k2.metric("% de victorias", "—" if pd.isna(j["club_win_rate"]) else f"{j['club_win_rate']*100:.0f}%")
    k3.metric("Puntos por partido", "—" if pd.isna(j["club_ppg"]) else f"{j['club_ppg']:.2f}")

# ---- Metodologia (siempre visible al final) ----
with st.expander("¿Cómo funciona y qué limitaciones tiene?"):
    mt = meta["metricas"]
    st.markdown(f"""
**Modelo:** XGBoost entrenado sobre {miles(mt['modelo_total']['n'])} jugadores activos con valor de mercado.
Cada predicción que ves la hizo un modelo que **no conocía a ese jugador** (validación cruzada de 5 folds).

**Precisión:**
- Ligas con estadísticas partido a partido: R² {mt['modelo_ligas_con_estadisticas']['r2_log']:.2f}, error mediano {mt['modelo_ligas_con_estadisticas']['error_pct_mediano']:.0f}%
- Ligas sin estadísticas: R² {mt['modelo_ligas_sin_estadisticas']['r2_log']:.2f}, error mediano {mt['modelo_ligas_sin_estadisticas']['error_pct_mediano']:.0f}%
- Referencia trivial (mediana por liga y posición): R² {mt['base_mediana_liga_posicion']['r2_log']:.2f}

**Limitaciones:**
- Solo 14 ligas europeas tienen estadísticas por partido en el dataset. En el resto el modelo usa perfil, contrato, club y experiencia previa.
- El contexto de club es el del club **actual**: si un jugador jugó cedido en otro equipo, sus partidos y su club actual no coinciden.
- Si un jugador llegó hace poco desde una liga sin estadísticas, solo se cuentan sus partidos en la liga cubierta.
- Los datos no se actualizan solos: corresponden al {fecha_linda(meta['fecha_ref'])}.
""")

# Para correr la app: streamlit run app.py en la terminal
