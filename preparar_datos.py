# -*- coding: utf-8 -*-
# ============================================================================
# preparar_datos.py
# ----------------------------------------------------------------------------
# Pipeline completo. Se corre UNA vez (o cada vez que se actualizan los CSV):
#   1. Construye los features con features.py
#   2. Evalua el modelo con validacion cruzada (5 folds)
#   3. Entrena el modelo final y lo guarda
#   4. Genera app_data.csv y metricas.json para la app
#
# Correr con:  python preparar_datos.py
# Requiere:    pip install -r requirements-pipeline.txt
# ============================================================================

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

import features as F

SEMILLA = 42
N_FOLDS = 5
PARAMS_XGB = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEMILLA,
    n_jobs=-1,
)


def nuevo_modelo():
    return XGBRegressor(**PARAMS_XGB)


def metricas(y_log, pred_log):
    """R² y MAE en escala log, y error porcentual mediano en euros."""
    real = np.expm1(y_log)
    pred = np.expm1(pred_log)
    return {
        "n": int(len(y_log)),
        "r2_log": round(float(r2_score(y_log, pred_log)), 4),
        "mae_log": round(float(mean_absolute_error(y_log, pred_log)), 4),
        "error_pct_mediano": round(float(np.median(np.abs(pred - real) / real) * 100), 1),
    }


def main():
    print("1/4 Construyendo features...")
    df, meta = F.construir_dataset(F.cargar_crudos())
    X = F.armar_X(df)

    tiene_valor = df["market_value_in_eur"].notna().to_numpy()
    y_log = np.log1p(df["market_value_in_eur"].to_numpy())
    cubierto = df["liga_con_estadisticas"].to_numpy()

    X_tr, y_tr = X[tiene_valor], y_log[tiene_valor]
    grupo_tr = (df.loc[tiene_valor, "liga_id"] + "|" + df.loc[tiene_valor, "position"]).to_numpy()

    print(f"   Jugadores activos: {len(df)} | con valor de mercado: {int(tiene_valor.sum())}")

    # ------------------------------------------------------------------------
    # 2. Validacion cruzada: cada jugador recibe una prediccion hecha por un
    #    modelo que NUNCA lo vio (out-of-fold). Antes prediciamos sobre los
    #    mismos datos de entrenamiento, y eso achica las diferencias reales.
    # ------------------------------------------------------------------------
    print(f"2/4 Validacion cruzada ({N_FOLDS} folds)...")
    oof = np.zeros(len(y_tr))
    oof_base = np.zeros(len(y_tr))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)

    for i, (a, b) in enumerate(kf.split(X_tr), start=1):
        m = nuevo_modelo()
        m.fit(X_tr.iloc[a], y_tr[a])
        oof[b] = m.predict(X_tr.iloc[b])

        # Linea de base simple: mediana del valor por liga + posicion.
        # Sirve para demostrar cuanto aporta el modelo por encima de algo trivial.
        med = pd.Series(y_tr[a]).groupby(grupo_tr[a]).median()
        oof_base[b] = pd.Series(grupo_tr[b]).map(med).fillna(np.median(y_tr[a])).to_numpy()
        print(f"   fold {i}/{N_FOLDS} listo")

    cub_tr = cubierto[tiene_valor]
    reporte = {
        "modelo_total": metricas(y_tr, oof),
        "modelo_ligas_con_estadisticas": metricas(y_tr[cub_tr], oof[cub_tr]),
        "modelo_ligas_sin_estadisticas": metricas(y_tr[~cub_tr], oof[~cub_tr]),
        "base_mediana_liga_posicion": metricas(y_tr, oof_base),
    }

    # Umbral de "diferencia significativa" = error tipico del modelo en ese grupo
    # (mediana del error absoluto en log). Si la diferencia es menor, es ruido.
    err = np.abs(oof - y_tr)
    umbral_cub = float(np.median(err[cub_tr]))
    umbral_nocub = float(np.median(err[~cub_tr]))

    # ------------------------------------------------------------------------
    # 3. Modelo final con todos los datos (para jugadores sin valor de mercado)
    # ------------------------------------------------------------------------
    print("3/4 Entrenando modelo final...")
    modelo = nuevo_modelo()
    modelo.fit(X_tr, y_tr)
    modelo.save_model(str(F.DATA_DIR / "modelo_transfermkt.json"))  # formato portable entre versiones

    pred_log = modelo.predict(X)                  # para quienes NO tienen valor
    pred_log[tiene_valor] = oof                   # para quienes SI: out-of-fold

    importancias = (
        pd.Series(modelo.feature_importances_, index=X.columns)
        .sort_values(ascending=False).head(15).round(4).to_dict()
    )

    # ------------------------------------------------------------------------
    # 4. Archivos para la app
    # ------------------------------------------------------------------------
    print("4/4 Guardando app_data.csv y metricas.json...")
    salida = df[[
        "player_id", "name", "club_nombre", "liga", "liga_id", "liga_con_estadisticas",
        "position", "sub_position", "foot", "height_in_cm", "age",
        "contract_expiration_date", "international_caps", "international_goals",
        "market_value_in_eur",
        "games_2y", "minutes_2y", "goals_2y", "assists_2y", "euro_games_2y",
        "minutes_per_game_2y",
        "career_games", "career_goals", "career_assists",
        "club_position", "club_win_rate", "club_ppg",
    ]].copy()
    salida["valor_predicho"] = np.expm1(pred_log).round(0)
    salida["umbral_log"] = np.where(cubierto, umbral_cub, umbral_nocub)
    salida["contract_expiration_date"] = salida["contract_expiration_date"].dt.strftime("%Y-%m-%d")
    salida["age"] = salida["age"].round(1)
    salida.to_csv(F.DATA_DIR / "app_data.csv", index=False)

    meta.update({
        "metricas": reporte,
        "umbral_log_con_estadisticas": round(umbral_cub, 4),
        "umbral_log_sin_estadisticas": round(umbral_nocub, 4),
        "importancias_top15": importancias,
        "features": list(X.columns),
        "params_xgb": PARAMS_XGB,
    })
    with open(F.DATA_DIR / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print()
    print(f"Fecha de corte de los datos: {meta['fecha_ref']}")
    for k, v in reporte.items():
        print(f"  {k:32s} R²={v['r2_log']:.3f}  MAE={v['mae_log']:.3f}  "
              f"error mediano={v['error_pct_mediano']}%  (n={v['n']})")
    print(f"  Umbral significativo: con estadisticas ±{(np.exp(umbral_cub)-1)*100:.0f}% "
          f"| sin estadisticas ±{(np.exp(umbral_nocub)-1)*100:.0f}%")
    print(f"Listo. app_data.csv con {len(salida)} jugadores.")


if __name__ == "__main__":
    main()
