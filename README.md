# Car Price Prediction — Production

Predicts used-car prices from 7 features via a single sklearn pipeline served by Flask.

## Layout

```
production/
├── car_price.ipynb      # EDA + model selection (imports the modules below)
├── preprocessing.py     # Data contract: feature lists + category cleaning (single source of truth)
├── train.py             # python train.py --data car_price_prediction.csv  → model/car_price_pipeline.pkl
├── app.py               # Flask app factory: / (form), /predict, /api/predict, /health
├── templates/index.html
├── tests/test_app.py    # pytest — trains a throwaway model, no data file needed
├── requirements.txt     # runtime only
├── requirements-dev.txt # + jupyter/plots/pytest
├── Dockerfile           # gunicorn, non-root, slim image
└── .dockerignore
```

## Quickstart

```bash
pip install -r requirements-dev.txt

# 1. Copy the dataset here, then train (~15 s, writes model/car_price_pipeline.pkl + metrics.json)
python train.py --data car_price_prediction.csv

# 2. Test
python -m pytest tests/

# 3. Serve
gunicorn -b 0.0.0.0:5000 "app:create_app()"

# or Docker
docker build -t car-price . && docker run -p 5000:5000 car-price
```

API:

```bash
curl -X POST localhost:5000/api/predict -H 'Content-Type: application/json' -d '{
  "Brand":"Toyota","Model":"Camry","Year":2018,"Fuel_Type":"Petrol",
  "Transmission":"Automatic","Owner_Type":"First","Kms_Driven":40000}'
# → {"predicted_price": 887131.96}
```

## Verified results (full dataset, 999,939 rows after dedup, 20% holdout)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| DecisionTree (pruned) | 361,636 | 640,140 | 0.688 |
| **HistGradientBoosting** (shipped) | **358,885** | 647,154 | 0.681 |
| Ridge | 616,023 | 1,191,110 | −0.081 |

HistGB is shipped: best MAE, more robust to unseen categories; the R² gap vs the tree is noise.
Artifact size: **162 KB** (old DecisionTree pickle: 106 MB). Old pipeline's honest R² was ≪ 0.37 (it was computed on a leaky, log-scale target).

## Audit of the original project — what was wrong

1. **`car_price_fixed.ipynb` cell 1** — `pd.read_csv('...csv')1`: stray `1` → `SyntaxError`. This is the error you hit.
2. **Broken deployment artifact** — the final pipeline in `car_price.ipynb` was trained on data where Brand/Fuel/Transmission were already label-encoded *integers* and Price was log1p-transformed. `app.py` fed it raw strings (OHE silently zeroed them out) and displayed the raw log output, e.g. "₹13.53".
3. **Data leakage** — imputation (cell 18), log transforms (29), label encoding (32) all fit on the full dataset before the train/test split.
4. **Swapped encoders** — `be/fe/te` were applied to Fuel/Transmission/Brand respectively, then dumped as `brand_encoder.pkl` (actually fuel), `fuel_encoder.pkl` (actually transmission), `transmission_encoder.pkl` (actually brand).
5. **Dirty categories never cleaned** — `Fuel_Type` had 11 raw values ('PETROL', ' Diesel', 'electrik', 'hybridd', …) for 5 real ones.
6. **Dead code** — 8-model comparison dict defined but never run; ~30 duplicate/diagnostic cells; double Horsepower imputation; deprecated `sns.distplot`.
7. **app.py** — `from sklearn import pipeline` shadowed by the loaded model, no validation, no error handling, `debug=True`, missing `templates/index.html`.
8. **Docker** — image included the 141 MB CSV, 320 MB of pickles, TensorFlow + Torch + Jupyter (~5 GB); Flask dev server as entrypoint.

## Files from the old project to DELETE

`car_price.ipynb` (old 80-cell), `car_price_fixed.ipynb`, `carprice.ipynb`, `app.py`, `test_mode.py`,
`car_price.pkl`, `car_price_pipeline.pkl` (old), `brand_encoder.pkl`, `fuel_encoder.pkl`,
`transmission_encoder.pkl`, `imputer.pkl`, `le.pkl`, old `requirements.txt`, old `Dockerfile`.
Keep only `car_price_prediction.csv` (copy it into this folder) and this `production/` directory.
