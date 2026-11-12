from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import make_cv_splits
from src.preprocessing import PreprocessingConfig, clean_training_data, load_data, split_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data():
    df, _ = clean_training_data(load_data(ROOT / "synthetic_data_208.csv"))
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def config() -> PreprocessingConfig:
    return PreprocessingConfig()


@pytest.fixture(scope="session")
def small_splits(data):
    """Three folds keep selection tests fast while exercising the real code path."""
    X_train = data[0]
    return make_cv_splits(X_train, n_splits=3, n_repeats=1)


def house(**overrides) -> pd.DataFrame:
    row = {"property_size": 2000.0, "number_of_bedrooms": 4.0, "number_of_bathrooms": 2.0,
           "location_type": "Suburban", "age_of_property": 10.0, "proximity_to_public_transport": 2.0,
           "crime_rate_in_area": 4.0, "school_quality_score": 9.0}
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.fixture
def make_house():
    return house


@pytest.fixture
def rng():
    return np.random.default_rng(0)
