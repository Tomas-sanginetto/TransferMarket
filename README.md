# Football Value Intelligence 🏆

ML model that predicts football player market values using Transfermarkt data, identifying undervalued and overvalued players based on performance, contract and club context.

**🔗 Live app: [football-value-intelligence.streamlit.app](https://football-value-intelligence.streamlit.app)**

## What it does

- Estimates the market value of **19,044 active players** across 32 leagues
- Flags a player as **undervalued** or **overvalued** only when the gap exceeds the model's typical error (otherwise: fair value)
- Shows every player's profile, contract, club form and career stats, plus recent match stats where the data exists
- Tells the user how reliable each prediction is, depending on data coverage for that league

## Results

All metrics are **out-of-fold** (5-fold cross-validation): each prediction is made by a model that never saw that player. R² and MAE on log(market value).

| Model | All players | Leagues with match stats | Leagues without match stats |
|---|---|---|---|
| Baseline: median by league + position | R² 0.39 | — | — |
| Previous version (v1 features) | R² 0.70 | R² 0.81 | R² 0.47 |
| **Current version (v2)** | **R² 0.78** | **R² 0.90** | **R² 0.56** |

Median error in euros: 32% in leagues with match stats, 48% in the rest.

> The v1 numbers above were re-computed on the same players with the same cross-validation, so the comparison is fair. The R² 0.696 originally reported for v1 is not comparable: it included ~20,000 retired players.

## What changed in v2 (data audit)

An audit of the app showed that many players displayed "0 matches". Root causes found and fixed:

1. **Coverage gap** — `appearances.csv` only has player-level match data for 14 European leagues. For the other 18 leagues (Argentina, Brazil, MLS, Japan, Austria…) it has none. v1 stored this as `0`, so the model confused *"no data"* with *"didn't play"*. v2 stores it as missing (XGBoost handles it natively) and adds an explicit coverage flag.
2. **Duplicate league names** — Germany's and Austria's leagues are both called `bundesliga` (same for Russia/Ukraine and Denmark/Romania). The league filter merged them. Now labelled with country.
3. **Retired players** — ~20,000 players had stopped playing years ago but were still shown with stale values (e.g. Xavi "at Barcelona"). Only active players (last season in the data) are kept.
4. **Shrinking time window** — "last 2 years" was computed from *today*, but the data ends on 2026-05-24, so the window shrank every day. It's now anchored to the last date in the data.
5. **In-sample predictions** — v1 predicted on its own training data, which hides real gaps. v2 uses out-of-fold predictions.
6. **Duplicate player names** — 440 active players share a name (e.g. three "Adama Traoré"). Selection is now by player ID.
7. **Coverage per club, not per league label** — relegated clubs (e.g. Swansea) are still tagged "Premier League" in the data; coverage is now decided by the matches the club actually played.

New features: contract years left, club form (table position, win rate, points per game — available for all 32 leagues), UEFA club competition matches, per-90 stats, minutes in the last year, exact age.

## Key findings (current model)

- Most important features: matches and minutes in the last 2 years. `international_caps`, which dominated v1 (~30% importance) and acted as a "already famous" shortcut, is no longer at the top.
- Among players worth €30M+, the model flags as **undervalued** players like Milos Kerkez (+93%), Viktor Gyökeres (+67%) and Dean Huijsen (+62%).
- It flags as **overvalued** players like Harry Kane (−73%) and Rodri (−71%). The model heavily penalizes age and missed matches (Rodri's 2024-25 injury), and it cannot see "star power".

## Limitations

- Player-level match stats exist for 14 leagues only. Elsewhere the model relies on profile, contract, club form and previous career.
- Club context is the player's **current** club. A player who spent the window on loan elsewhere gets his parent club's context.
- A player who recently moved from an uncovered league only has his matches in the covered league counted.
- Data is a snapshot (2026-05-24), not live.

## Data

- Source: [Transfermarkt dataset on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores)
- 19,044 active players (17,646 with a market value), 32 leagues
- 1,887,171 match appearances (2012–2026) for the 14 covered leagues

## Project structure

```
features.py              → Single source of truth: builds every feature (used by training and app data)
preparar_datos.py        → Full pipeline: features → cross-validation → final model → app data
app.py                   → Streamlit app (only reads precomputed files, no model at runtime)
app_data.csv             → Precomputed predictions and player info for the app
metricas.json            → Metrics, thresholds, reference dates, feature list
modelo_transfermkt.json  → Trained XGBoost model (portable format)
01_exploracion.ipynb     → Original EDA (v1, kept for history)
02_modelo.ipynb          → Original model notebook (v1, kept for history)
```

To rebuild: `pip install -r requirements-pipeline.txt` then `python preparar_datos.py`.
To run the app locally: `pip install -r requirements.txt` then `streamlit run app.py`.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · Streamlit · Streamlit Community Cloud

## Roadmap

- [x] Phase 1 — EDA, cleaning, model training
- [x] Phase 2 — Web app deployed (Streamlit Cloud)
- [x] Data audit and model v2
- [ ] Explain each prediction (SHAP)
- [ ] Backtesting: did "undervalued" players actually rise in value?
- [ ] Phase 3 — Automated data updates

## Dev log

| Date | Progress |
|---|---|
| Jun 25, 2026 | Dataset downloaded, EDA completed |
| Jun 26, 2026 | Data cleaning, first Random Forest model — R² 0.41 |
| Jun 27, 2026 | Added appearance stats, XGBoost — R² improved to 0.696 |
| Jul 2, 2026 | Streamlit web app — player search with market value display |
| Jul 22, 2026 | App shows model prediction and under/overvalued verdict |
| Sep 25, 2026 | Precomputed data pipeline, league filter, app deployed online |
| Sep 27, 2026 | Data audit: 7 issues found and fixed. Model v2 — R² 0.78 (out-of-fold) |
