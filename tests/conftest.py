import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture()
def sample_df():
    """Minimal Heart Disease-like DataFrame for unit tests."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "age": np.random.randint(30, 75, n),
            "sex": np.random.randint(0, 2, n),
            "cp": np.random.randint(0, 4, n),
            "trestbps": np.random.randint(90, 200, n),
            "chol": np.random.randint(120, 400, n),
            "fbs": np.random.randint(0, 2, n),
            "restecg": np.random.randint(0, 3, n),
            "thalach": np.random.randint(70, 210, n),
            "exang": np.random.randint(0, 2, n),
            "oldpeak": np.random.uniform(0, 6, n).round(1),
            "slope": np.random.randint(0, 3, n),
            "ca": np.random.randint(0, 4, n),
            "thal": np.random.randint(0, 3, n),
            "target": np.random.randint(0, 2, n),
        }
    )


@pytest.fixture()
def xy_scaled(sample_df):
    """Return scaled train/test arrays from sample data."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = sample_df.drop("target", axis=1)
    y = sample_df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train, X_test, X_train_s, X_test_s, y_train, y_test
