# Football Value Intelligence 🏆

ML model that predicts football player market values using Transfermarkt historical data, identifying undervalued and overvalued players based on performance statistics.

## What it does

- Predicts the market value of any football player in euros
- Detects **undervalued players** (model predicts higher value than Transfermarkt)
- Detects **overvalued players** (model predicts lower value than Transfermarkt)

## Results

| Model | R² | MAE |
|---|---|---|
| Random Forest (basic) | 0.413 | 0.920 |
| Random Forest + career stats | 0.659 | 0.709 |
| Random Forest + recent stats (2y) | 0.671 | 0.692 |
| **XGBoost + recent stats (2y)** | **0.696** | **0.664** |

## Key findings

- **Most undervalued:** Lamine Yamal (€200M real → model predicts €316M) — later confirmed by CIES at €358M
- **Most overvalued:** Pedri, Dembélé — penalized for injury-heavy recent seasons
- The model correctly identifies players whose recent performance justifies a higher market value

## Data

- Source: [Transfermarkt dataset on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores)
- 48,317 players across all major leagues
- 1,887,171 match appearances with goals, assists and minutes played

## Project structure

```
01_Exploracion.ipynb   → EDA and data cleaning
02_Modelo.ipynb        → Feature engineering, model training and evaluation
modelo_transfermkt.pkl → Trained XGBoost model
```

## Tech stack

Python · pandas · scikit-learn · XGBoost · matplotlib

## Roadmap

- [x] Phase 1 — EDA, cleaning, model training
- [x] Phase 2 — Web interface (Streamlit) — in progress
- [ ] Phase 3 — Automated weekly data updates

## Dev log

| Date | Progress |
|---|---|
| Jun 25, 2026 | Dataset downloaded, EDA completed |
| Jun 26, 2026 | Data cleaning, first Random Forest model (R² 0.41) |
| Jun 27, 2026 | Added appearance stats, XGBoost — R² improved to 0.696 |
| Jul 2, 2026 | Streamlit web app — player search with market value display |
| Jul 22, 2026 | Streamlit app now shows model prediction and under/overvalued verdict |
