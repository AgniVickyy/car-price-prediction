"""Flask service for car price prediction.

Run (dev):   flask --app "app:create_app()" run
Run (prod):  gunicorn -b 0.0.0.0:5000 "app:create_app()"
"""

import os
from datetime import date

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from preprocessing import FEATURES, clean

MAX_YEAR = date.today().year + 1


def _parse_payload(data: dict) -> pd.DataFrame:
    """Validate one prediction request. Raises ValueError with a user message."""
    missing = [f for f in FEATURES if data.get(f) in (None, "")]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    try:
        year = int(data["Year"])
        kms = int(data["Kms_Driven"])
    except (TypeError, ValueError):
        raise ValueError("Year and Kms_Driven must be integers")
    if not 1980 <= year <= MAX_YEAR:
        raise ValueError(f"Year must be between 1980 and {MAX_YEAR}")
    if not 0 <= kms <= 5_000_000:
        raise ValueError("Kms_Driven must be between 0 and 5,000,000")

    row = {f: str(data[f]) for f in FEATURES}
    row["Year"], row["Kms_Driven"] = year, kms
    return clean(pd.DataFrame([row]))


def create_app(model_path: str | None = None) -> Flask:
    app = Flask(__name__)
    path = model_path or os.environ.get("MODEL_PATH", "model/car_price_pipeline.pkl")
    pipeline = joblib.load(path)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.post("/predict")
    def predict_form():
        try:
            X = _parse_payload(request.form.to_dict())
        except ValueError as exc:
            return render_template("index.html", error=str(exc)), 400
        price = float(pipeline.predict(X)[0])
        return render_template("index.html", prediction=f"Predicted Price: ₹{price:,.0f}")

    @app.post("/api/predict")
    def predict_api():
        try:
            X = _parse_payload(request.get_json(silent=True) or {})
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(predicted_price=round(float(pipeline.predict(X)[0]), 2))

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000)
