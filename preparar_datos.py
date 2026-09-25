# -*- coding: utf-8 -*-
# ============================================================================
# preparar_datos.py
# ----------------------------------------------------------------------------
# Este script se corre UNA sola vez (o cada vez que actualizamos los datos).
# Hace todo el trabajo pesado: reconstruye los features, corre el modelo, y
# guarda un unico archivo chico (app_data.csv) con todo lo que la app necesita.
#
# Asi la app deployada NO tiene que cargar appearances.csv (148 MB) ni el modelo.
# Solo lee app_data.csv, que pesa pocos MB. Queda rapida y se puede subir a GitHub.
#
# Correr con:  python preparar_datos.py
# ============================================================================

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

print("Cargando datos...")
df_players = pd.read_csv("players.csv")
df_apps = pd.read_csv("appearances.csv")
df_competitions = pd.read_csv("competitions.csv")
modelo = joblib.load("modelo_transfermkt.pkl")

print("Construyendo features (mismo pipeline que el notebook 02)...")
df = df_players.copy()

# Edad (igual que en el entrenamiento: 2024 como año de referencia)
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
df['age'] = 2024 - df['date_of_birth'].dt.year

# Estadisticas de TODA la carrera
stats_carrera = df_apps.groupby('player_id').agg(
    total_goals=('goals', 'sum'),
    total_assists=('assists', 'sum'),
    total_minutes=('minutes_played', 'sum'),
    total_games=('appearance_id', 'count')
).reset_index()
df = df.merge(stats_carrera, on='player_id', how='left')
cols_carrera = ['total_goals', 'total_assists', 'total_minutes', 'total_games']
df[cols_carrera] = df[cols_carrera].fillna(0)

# Columnas utiles (mismo orden que en el notebook)
columnas = [
    'player_id', 'market_value_in_eur', 'age', 'position', 'sub_position',
    'height_in_cm', 'foot', 'international_caps', 'international_goals',
    'current_club_domestic_competition_id',
    'total_goals', 'total_assists', 'total_minutes', 'total_games'
]
df = df[columnas]

# Misma limpieza
df = df.dropna(subset=['market_value_in_eur', 'age', 'sub_position',
                       'current_club_domestic_competition_id'])
df['height_in_cm'] = df['height_in_cm'].fillna(df['height_in_cm'].median())
df['foot'] = df['foot'].fillna('right')
df['international_caps'] = df['international_caps'].fillna(0)
df['international_goals'] = df['international_goals'].fillna(0)
df = df[(df['age'] >= 15) & (df['age'] <= 45)]

# Guardamos los ids que sobrevivieron la limpieza (para juntar datos despues)
ids_limpios = df['player_id'].copy()

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['position', 'sub_position', 'foot',
                                         'current_club_domestic_competition_id'])

# Estadisticas de los ULTIMOS 2 AÑOS (dinamico desde hoy)
df_apps['date'] = pd.to_datetime(df_apps['date'])
hace_2_anios = pd.Timestamp(datetime.now()) - pd.DateOffset(years=2)
df_recent = df_apps[df_apps['date'] >= hace_2_anios]

stats_recent = df_recent.groupby('player_id').agg(
    goals_2y=('goals', 'sum'),
    assists_2y=('assists', 'sum'),
    minutes_2y=('minutes_played', 'sum'),
    games_2y=('appearance_id', 'count')
).reset_index()
stats_recent['goals_per_game'] = stats_recent['goals_2y'] / stats_recent['games_2y']
stats_recent['assists_per_game'] = stats_recent['assists_2y'] / stats_recent['games_2y']
stats_recent['minutes_per_game'] = stats_recent['minutes_2y'] / stats_recent['games_2y']

df_encoded = df_encoded.merge(stats_recent, on='player_id', how='left')
cols_recientes = ['goals_2y', 'assists_2y', 'minutes_2y', 'games_2y',
                  'goals_per_game', 'assists_per_game', 'minutes_per_game']
df_encoded[cols_recientes] = df_encoded[cols_recientes].fillna(0)

print("Prediciendo con el modelo...")
X = df_encoded.drop(columns=['market_value_in_eur', 'player_id'])

# Alineamos columnas con lo que el modelo espera
if hasattr(modelo, 'feature_names_in_'):
    X = X.reindex(columns=modelo.feature_names_in_, fill_value=0)

# expm1 deshace el log1p → euros reales
predicciones = np.expm1(modelo.predict(X))

print("Armando el archivo final...")
# Diccionario codigo_liga -> nombre_liga
mapa_ligas = dict(zip(df_competitions['competition_id'], df_competitions['name']))

# Traemos los datos "lindos" (nombre, club) desde players.csv original
info = df_players.set_index('player_id')

app_data = pd.DataFrame({
    'player_id': df_encoded['player_id'].values,
    'valor_predicho': predicciones,
})
app_data['name'] = app_data['player_id'].map(info['name'])
app_data['current_club_name'] = app_data['player_id'].map(info['current_club_name'])
app_data['position'] = app_data['player_id'].map(info['position'])
app_data['market_value_in_eur'] = app_data['player_id'].map(info['market_value_in_eur'])
app_data['competition_id'] = app_data['player_id'].map(info['current_club_domestic_competition_id'])
app_data['liga'] = app_data['competition_id'].map(mapa_ligas)

# Estadisticas de 2 años para mostrar en la ficha del jugador
stats_idx = stats_recent.set_index('player_id')
for c in ['games_2y', 'goals_2y', 'assists_2y', 'minutes_per_game']:
    app_data[c] = app_data['player_id'].map(stats_idx[c]).fillna(0)

# Guardamos. Formato parquet: mas chico y rapido que csv (requiere pyarrow)
app_data.to_csv("app_data.csv", index=False)
print(f"Listo. app_data.csv guardado con {len(app_data)} jugadores.")
print(f"Tamaño aproximado en memoria: {app_data.memory_usage(deep=True).sum() / 1e6:.1f} MB")
