"""API tests. Trains a tiny throwaway model so tests need no data file."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402
from train import build_pipeline  # noqa: E402
from preprocessing import FEATURES, TARGET, clean  # noqa: E402

VALID = {
    "Brand": "Toyota", "Model": "Camry", "Year": 2018, "Fuel_Type": "Petrol",
    "Transmission": "Manual", "Owner_Type": "First", "Kms_Driven": 40000,
}


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "Brand": rng.choice(["Toyota", "Honda", "BMW"], n),
        "Model": rng.choice(["Camry", "Civic", "X1"], n),
        "Year": rng.integers(2000, 2025, n),
        "Fuel_Type": rng.choice(["Petrol", "diesel ", "HYBRID"], n),
        "Transmission": rng.choice(["Manual", "Automatic"], n),
        "Owner_Type": rng.choice(["First", "Second"], n),
        "Kms_Driven": rng.integers(1000, 300000, n),
        TARGET: rng.uniform(1e5, 3e6, n),
    })
    df = clean(df)
    pipe = build_pipeline(random_state=0).fit(df[FEATURES], df[TARGET])
    path = tmp_path_factory.mktemp("model") / "pipe.pkl"
    joblib.dump(pipe, path)
    app = create_app(model_path=str(path))
    app.testing = True
    return app.test_client()


def test_health(client):
    assert client.get("/health").json == {"status": "ok"}


def test_api_predict_ok(client):
    r = client.post("/api/predict", json=VALID)
    assert r.status_code == 200
    assert r.json["predicted_price"] > 0


def test_api_predict_handles_dirty_and_unknown_categories(client):
    r = client.post("/api/predict", json={**VALID, "Fuel_Type": " PETROL ", "Brand": "Lada"})
    assert r.status_code == 200
    assert r.json["predicted_price"] > 0


def test_api_missing_field(client):
    r = client.post("/api/predict", json={k: v for k, v in VALID.items() if k != "Brand"})
    assert r.status_code == 400
    assert "Brand" in r.json["error"]


def test_api_invalid_year(client):
    assert client.post("/api/predict", json={**VALID, "Year": 1900}).status_code == 400


def test_form_predict(client):
    r = client.post("/predict", data={k: str(v) for k, v in VALID.items()})
    assert r.status_code == 200
    assert b"Predicted Price" in r.data
