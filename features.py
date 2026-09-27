# -*- coding: utf-8 -*-
# ============================================================================
# features.py
# ----------------------------------------------------------------------------
# UNICA fuente de verdad para construir los datos del modelo.
# Lo usa preparar_datos.py (entrenamiento + datos de la app).
# Tener todo en un solo lugar evita que el entrenamiento y la app calculen
# las cosas distinto (eso fue lo que nos paso con la ventana de 2 años).
# ============================================================================

from pathlib import Path

import numpy as np
import pandas as pd

# Carpeta donde estan los CSV (la misma carpeta que este archivo)
DATA_DIR = Path(__file__).resolve().parent

VENTANA_ANIOS = 2        # "ultimos 2 años" para las estadisticas recientes
LIGA_OTRA = "OTRA"       # jugadores cuyo club no esta en ninguna de las 32 ligas del dataset

# Ligas que aparecen en los datos pero NO en competitions.csv (sin nombre ni pais)
NOMBRES_FALTANTES = {"COL1": "Primera A (Colombia)"}

# Columnas categoricas que se convierten a 0/1 (one-hot)
CATEGORICAS = ["position", "sub_position", "foot", "liga_id"]


# ----------------------------------------------------------------------------
# 1. CARGA DE ARCHIVOS
# ----------------------------------------------------------------------------
def cargar_crudos(data_dir: Path = DATA_DIR) -> dict:
    players = pd.read_csv(
        data_dir / "players.csv",
        parse_dates=["date_of_birth", "contract_expiration_date"],
    )
    apps = pd.read_csv(
        data_dir / "appearances.csv",
        usecols=["player_id", "game_id", "date", "competition_id", "goals",
                 "assists", "minutes_played", "yellow_cards", "red_cards"],
        parse_dates=["date"],
    )
    comps = pd.read_csv(data_dir / "competitions.csv")
    games = pd.read_csv(
        data_dir / "games.csv",
        usecols=["game_id", "competition_id", "date", "competition_type"],
        parse_dates=["date"],
    )
    club_games = pd.read_csv(
        data_dir / "club_games.csv",
        usecols=["game_id", "club_id", "own_goals", "opponent_goals", "own_position"],
    )
    return {"players": players, "apps": apps, "comps": comps,
            "games": games, "club_games": club_games}


# ----------------------------------------------------------------------------
# 2. HELPERS
# ----------------------------------------------------------------------------
def _division_segura(a, b):
    """a / b, pero devuelve 0 cuando b es 0 (ej: goles por 90 de alguien con 0 minutos)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros_like(a)
    np.divide(a, b, out=out, where=(b > 0))
    # Si a o b son NaN (sin datos), el resultado tiene que seguir siendo NaN
    out[np.isnan(a) | np.isnan(b)] = np.nan
    return out


def _etiqueta_liga(nombre: str, pais) -> str:
    """'premier-league' + 'England' -> 'Premier League (England)'."""
    lindo = str(nombre).replace("-", " ").title()
    return f"{lindo} ({pais})" if pd.notna(pais) else lindo


# ----------------------------------------------------------------------------
# 3. CONSTRUCCION DEL DATASET
# ----------------------------------------------------------------------------
def construir_dataset(crudos: dict):
    """
    Devuelve (df, meta):
      df   -> una fila por jugador ACTIVO con features, datos para mostrar y target
      meta -> informacion de referencia (fecha de corte, ligas cubiertas, etc.)
    """
    players = crudos["players"]
    apps = crudos["apps"]
    comps = crudos["comps"]
    games = crudos["games"]
    club_games = crudos["club_games"]

    # --- Fecha de referencia: la ULTIMA fecha con datos, no "hoy" ------------
    # Antes usabamos datetime.now(): como el dataset termina en mayo 2026,
    # la ventana de 2 años se iba achicando sola dia a dia.
    fecha_ref = apps["date"].max().normalize()
    inicio_ventana = fecha_ref - pd.DateOffset(years=VENTANA_ANIOS)
    inicio_ultimo_anio = fecha_ref - pd.DateOffset(years=1)

    # --- Solo jugadores ACTIVOS ----------------------------------------------
    # last_season = ultima temporada registrada. Los que no jugaron la ultima
    # temporada estan retirados o fuera del dataset (Xavi, Iniesta, etc.)
    ultima_temporada = int(players["last_season"].max())
    df = players[players["last_season"] == ultima_temporada].copy()

    # --- Liga actual ------------------------------------------------------------
    df["liga_id"] = df["current_club_domestic_competition_id"].fillna(LIGA_OTRA)

    ligas_domesticas = comps[comps["type"] == "domestic_league"]
    info_ligas = ligas_domesticas.set_index("competition_id")[["name", "country_name"]]

    # Ids de liga domestica: las de competitions.csv + las que usan los jugadores
    # (COL1 no esta en competitions.csv, y sus partidos vienen sin tipo)
    ids_ligas = set(ligas_domesticas["competition_id"]) | set(
        players["current_club_domestic_competition_id"].dropna()
    )

    # Ligas con estadisticas de jugadores: las que tienen partidos en appearances
    # dentro de la ventana. Se calcula desde los datos, no se escribe a mano.
    apps_ventana = apps[(apps["date"] > inicio_ventana) & (apps["date"] <= fecha_ref)]
    ligas_con_datos = sorted(set(apps_ventana["competition_id"]) & ids_ligas)

    # Etiqueta unica para mostrar (arregla "bundesliga" Alemania vs Austria)
    etiquetas = {
        cid: _etiqueta_liga(row["name"], row["country_name"])
        for cid, row in info_ligas.iterrows()
    }
    etiquetas.update(NOMBRES_FALTANTES)
    etiquetas[LIGA_OTRA] = "Otras ligas (fuera del dataset)"
    df["liga"] = df["liga_id"].map(etiquetas).fillna("Liga " + df["liga_id"])

    # Partidos de liga domestica con fecha (sirve para cobertura y contexto del club)
    partidos_liga = games[
        (games["competition_type"] == "domestic_league")
        | games["competition_id"].isin(ids_ligas)
    ][["game_id", "competition_id", "date"]]
    cg_liga = club_games.merge(partidos_liga, on="game_id", how="inner")

    # Cobertura POR CLUB: el club jugo partidos de una liga con estadisticas
    # en el ultimo año. Asi un club descendido que la ficha todavia marca como
    # "Premier League" (ej: Swansea) no aparece con 0 partidos falsos.
    clubes_cubiertos = set(
        cg_liga.loc[
            cg_liga["competition_id"].isin(ligas_con_datos)
            & (cg_liga["date"] > inicio_ultimo_anio)
            & (cg_liga["date"] <= fecha_ref),
            "club_id",
        ]
    )
    df["liga_con_estadisticas"] = df["current_club_id"].isin(clubes_cubiertos)

    # --- Datos personales -----------------------------------------------------
    # Edad exacta a la fecha de referencia (antes era 2024 - año de nacimiento)
    df["age"] = (fecha_ref - df["date_of_birth"]).dt.days / 365.25

    # Años de contrato restantes. Si ya vencio, 0. Si no hay dato, queda NaN.
    df["contract_years_left"] = (
        (df["contract_expiration_date"] - fecha_ref).dt.days / 365.25
    ).clip(lower=0)

    df["international_caps"] = df["international_caps"].fillna(0)
    df["international_goals"] = df["international_goals"].fillna(0)
    df["sub_position"] = df["sub_position"].fillna("Missing")
    df["foot"] = df["foot"].fillna("unknown")
    # height_in_cm queda con NaN si falta: XGBoost sabe manejar NaN

    # --- Competiciones europeas de clubes -----------------------------------
    es_europea = (
        comps["sub_type"].astype(str).str.startswith("uefa_")
        & (comps["type"] != "national_team_competition")
    )
    ids_europeas = set(comps.loc[es_europea, "competition_id"])

    # --- Estadisticas individuales: ultimos 2 años ---------------------------
    g2 = apps_ventana.groupby("player_id").agg(
        games_2y=("game_id", "count"),
        minutes_2y=("minutes_played", "sum"),
        goals_2y=("goals", "sum"),
        assists_2y=("assists", "sum"),
        yellow_2y=("yellow_cards", "sum"),
        red_2y=("red_cards", "sum"),
    )
    g2["euro_games_2y"] = (
        apps_ventana[apps_ventana["competition_id"].isin(ids_europeas)]
        .groupby("player_id")["game_id"].count()
    )
    g1 = (
        apps[(apps["date"] > inicio_ultimo_anio) & (apps["date"] <= fecha_ref)]
        .groupby("player_id")["minutes_played"].sum()
        .rename("minutes_1y")
    )
    recientes = g2.join(g1, how="outer")
    cols_recientes = list(recientes.columns)
    df = df.merge(recientes, left_on="player_id", right_index=True, how="left")

    # CLAVE: separar "no jugo" de "no tenemos datos"
    #  - Liga CON estadisticas y sin partidos -> realmente no jugo -> 0
    #  - Liga SIN estadisticas -> no sabemos -> NaN (XGBoost lo trata distinto)
    cubierto = df["liga_con_estadisticas"]
    df.loc[cubierto, cols_recientes] = df.loc[cubierto, cols_recientes].fillna(0)
    df.loc[~cubierto, cols_recientes] = np.nan

    df["minutes_per_game_2y"] = _division_segura(df["minutes_2y"], df["games_2y"])
    df["goals_per90_2y"] = _division_segura(df["goals_2y"] * 90, df["minutes_2y"])
    df["assists_per90_2y"] = _division_segura(df["assists_2y"] * 90, df["minutes_2y"])

    # --- Experiencia en competiciones cubiertas (toda la carrera) ------------
    # Esto SI vale para todos: un jugador de la MLS que antes jugo en la
    # Premier tiene esos partidos registrados. 0 = nunca jugo en esas ligas.
    carrera = apps[apps["date"] <= fecha_ref].groupby("player_id").agg(
        career_games=("game_id", "count"),
        career_minutes=("minutes_played", "sum"),
        career_goals=("goals", "sum"),
        career_assists=("assists", "sum"),
    )
    df = df.merge(carrera, left_on="player_id", right_index=True, how="left")
    cols_carrera = list(carrera.columns)
    df[cols_carrera] = df[cols_carrera].fillna(0)

    # --- Contexto del club (existe para las 32 ligas) ------------------------
    cg = cg_liga[(cg_liga["date"] > inicio_ventana) & (cg_liga["date"] <= fecha_ref)].copy()
    cg["puntos"] = np.select(
        [cg["own_goals"] > cg["opponent_goals"], cg["own_goals"] == cg["opponent_goals"]],
        [3, 1], default=0,
    )
    cg["gano"] = (cg["own_goals"] > cg["opponent_goals"]).astype(int)
    cg["dif_gol"] = cg["own_goals"] - cg["opponent_goals"]

    club = cg.groupby("club_id").agg(
        club_games_2y=("game_id", "count"),
        club_ppg=("puntos", "mean"),
        club_win_rate=("gano", "mean"),
        club_gd_per_game=("dif_gol", "mean"),
    )
    # Posicion en la tabla en el ultimo partido de liga que jugo el club
    ultima_pos = (
        cg.dropna(subset=["own_position"])
        .sort_values("date")
        .groupby("club_id")["own_position"].last()
        .rename("club_position")
    )
    club = club.join(ultima_pos, how="left")
    df = df.merge(club, left_on="current_club_id", right_index=True, how="left")

    # --- Datos para mostrar ---------------------------------------------------
    df["club_nombre"] = df["current_club_name"].fillna("Club fuera del dataset")

    meta = {
        "fecha_ref": fecha_ref.strftime("%Y-%m-%d"),
        "inicio_ventana": inicio_ventana.strftime("%Y-%m-%d"),
        "ultima_temporada": ultima_temporada,
        "ligas_con_estadisticas": ligas_con_datos,
        "jugadores_activos": int(len(df)),
    }
    return df.reset_index(drop=True), meta


# ----------------------------------------------------------------------------
# 4. MATRIZ DE FEATURES PARA EL MODELO
# ----------------------------------------------------------------------------
FEATURES_NUMERICAS = [
    # Personales
    "age", "height_in_cm", "contract_years_left",
    "international_caps", "international_goals",
    # Rendimiento reciente (NaN si la liga no tiene estadisticas)
    "games_2y", "minutes_2y", "goals_2y", "assists_2y", "yellow_2y", "red_2y",
    "euro_games_2y", "minutes_1y",
    "minutes_per_game_2y", "goals_per90_2y", "assists_per90_2y",
    # Experiencia en competiciones cubiertas
    "career_games", "career_minutes", "career_goals", "career_assists",
    # Contexto del club
    "club_games_2y", "club_ppg", "club_win_rate", "club_gd_per_game", "club_position",
    # Indicador explicito de cobertura
    "liga_con_estadisticas",
]


def armar_X(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve la matriz numerica que recibe el modelo."""
    X = df[FEATURES_NUMERICAS].astype(float)
    dummies = pd.get_dummies(df[CATEGORICAS].astype(str), dtype=float)
    X = pd.concat([X, dummies], axis=1)
    # Orden fijo de columnas para que sea reproducible
    return X.reindex(sorted(X.columns), axis=1)
