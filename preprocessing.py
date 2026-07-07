"""Shared data contract and cleaning logic for training and inference.

Single source of truth: train.py, app.py and the notebook all import from here,
so the model and the API can never disagree about features or category spelling.
"""

import pandas as pd

TARGET = "Price"
CATEGORICAL = ["Brand", "Model", "Fuel_Type", "Transmission", "Owner_Type"]
NUMERIC = ["Year", "Kms_Driven"]
FEATURES = CATEGORICAL + NUMERIC

# Raw data contains typos and case/whitespace variants ('PETROL', ' Diesel',
# 'electrik', 'hybridd', ...). Map everything to canonical labels.
_FUEL_MAP = {
    "petrol": "Petrol",
    "diesel": "Diesel",
    "cng": "CNG",
    "electric": "Electric",
    "electrik": "Electric",
    "hybrid": "Hybrid",
    "hybridd": "Hybrid",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical values. Safe on frames with or without TARGET."""
    df = df.copy()
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].str.strip()
    if "Fuel_Type" in df.columns:
        df["Fuel_Type"] = df["Fuel_Type"].str.lower().map(_FUEL_MAP)
    return df
