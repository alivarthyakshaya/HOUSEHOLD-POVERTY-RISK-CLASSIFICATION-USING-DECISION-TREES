import numpy as np

from src.data import add_features, clean
from src.generate_synthetic_data import generate
from src.models import baseline_tree


def test_synthetic_shape_and_classes():
    df = generate(500, seed=1)
    assert df.shape[0] == 500
    assert set(df["Target"].unique()) <= {1, 2, 3, 4}


def test_clean_returns_numeric_no_nan():
    X, y = clean(generate(500, seed=1))
    assert len(X) == len(y) == 500
    assert not X.isna().any().any()
    assert all(np.issubdtype(t, np.number) for t in X.dtypes)
    assert "Target" not in X.columns and "Id" not in X.columns


def test_features_added():
    X, _ = clean(generate(300, seed=2))
    X = add_features(X)
    assert {"people_per_room", "asset_score"} <= set(X.columns)


def test_tree_fits_and_predicts():
    X, y = clean(generate(400, seed=3))
    model = baseline_tree().fit(X, y)
    assert set(model.predict(X)) <= {1, 2, 3, 4}
