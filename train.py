"""Train the car-price model and export one self-contained pipeline artifact.

Usage:
    python train.py --data car_price_prediction.csv --output-dir model
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from preprocessing import CATEGORICAL, FEATURES, NUMERIC, TARGET, clean

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_pipeline(random_state: int) -> Pipeline:
    """Impute/encode inside the pipeline -> no leakage, one deployable artifact."""
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                CATEGORICAL,
            ),
            ("num", "passthrough", NUMERIC),
        ]
    )
    model = HistGradientBoostingRegressor(
        categorical_features=list(range(len(CATEGORICAL))),
        random_state=random_state,
    )
    # Train on log1p(Price), invert with expm1 -> metrics in currency units.
    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=model, func=np.log1p, inverse_func=np.expm1
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="car_price_prediction.csv")
    parser.add_argument("--output-dir", default="model")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data, usecols=FEATURES + [TARGET])
    df = clean(df.drop_duplicates().reset_index(drop=True))
    log.info("Loaded %d rows", len(df))

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    pipe = build_pipeline(args.random_state)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
        "R2": float(r2_score(y_test, pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": FEATURES,
    }
    log.info("Test metrics: %s", metrics)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "car_price_pipeline.pkl", compress=3)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Saved %s", out / "car_price_pipeline.pkl")


if __name__ == "__main__":
    main()
