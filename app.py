import streamlit as st # "st" es el alias de streamlit, pura comodidad
import pandas as pd # "pd" es el alias de pandas

# Configuracion de la pagina (tiene que ir antes que cualquier otro st.)
st.set_page_config(page_title="Football Value Intelligence", page_icon="⚽")

# ----------------------------------------------------------------------------
# La app ya NO reconstruye los features ni corre el modelo.
# Todo eso se pre-calcula con preparar_datos.py y queda en app_data.csv (chico).
# Aca solo leemos ese archivo → la app arranca rapido y se puede deployar.
# ----------------------------------------------------------------------------

@st.cache_data
def cargar_datos():
    return pd.read_csv("app_data.csv")

df = cargar_datos()

# Funcion para formatear euros con puntos de miles y el simbolo adelante
def formato_euros(valor):
    return "€" + "{:,}".format(int(valor)).replace(",", ".")

# ---------------- INTERFAZ ----------------

st.title("⚽ Football Value Intelligence")
st.subheader("Detectá jugadores subvalorados y sobrevalorados")

# ---- Filtro por liga ----
ligas_disponibles = sorted(df['liga'].dropna().unique().tolist())

liga = st.selectbox(
    "Filtrá por liga (opcional):",
    options=["Todas las ligas"] + ligas_disponibles
)

# Si el usuario elige una liga, nos quedamos solo con los jugadores de esa liga
if liga == "Todas las ligas":
    df_filtrado = df
else:
    df_filtrado = df[df['liga'] == liga]

# Lista de nombres para el autocomplete (ya filtrada por liga si corresponde)
nombres_disponibles = sorted(df_filtrado['name'].dropna().unique().tolist())

nombre = st.selectbox(
    "Buscá un jugador:",
    options=[""] + nombres_disponibles,
    placeholder="Escribí un nombre..."
)

if nombre: # Solo se ejecuta si el usuario selecciono algo
    jugador = df_filtrado[df_filtrado['name'] == nombre].iloc[0]

    # Info basica del jugador
    st.markdown(f"### {jugador['name']}")
    st.write(f"{jugador['current_club_name']} · {jugador['position']}")

    valor_real = jugador['market_value_in_eur']
    valor_predicho = jugador['valor_predicho']
    diferencia = valor_predicho - valor_real

    # Mostramos los dos valores lado a lado en columnas
    col1, col2 = st.columns(2)
    col1.metric("Valor Transfermarkt", formato_euros(valor_real))
    col2.metric("Valor según el modelo", formato_euros(valor_predicho))

    # El porcentaje nos dice que tan grande es la diferencia en relacion al valor real
    porcentaje = (diferencia / valor_real) * 100

    # Umbral del 20% para no marcar diferencias chicas como significativas
    if porcentaje > 20:
        st.success(f"📈 **SUBVALORADO** — el modelo lo valúa {porcentaje:.0f}% "
                   f"por encima ({formato_euros(diferencia)} más)")
    elif porcentaje < -20:
        st.error(f"📉 **SOBREVALORADO** — el modelo lo valúa {abs(porcentaje):.0f}% "
                 f"por debajo ({formato_euros(abs(diferencia))} menos)")
    else:
        st.info(f"✅ **VALOR JUSTO** — la diferencia es de solo {porcentaje:.0f}%")

    # Estadisticas en las que se baso el modelo
    st.divider()
    st.markdown("**Estadísticas de los últimos 2 años**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partidos", int(jugador['games_2y']))
    c2.metric("Goles", int(jugador['goals_2y']))
    c3.metric("Asistencias", int(jugador['assists_2y']))
    c4.metric("Min/partido", f"{jugador['minutes_per_game']:.0f}")

# Para correr la app: streamlit run app.py en la terminal
