"""Score new households with the saved tree.

    python predict.py path/to/new_households.csv
"""
import sys

import joblib
import pandas as pd

from src.data import TARGET_NAMES, add_features, clean

if __name__ == "__main__":
    bundle = joblib.load("outputs/decision_tree_model.joblib")
    df = pd.read_csv(sys.argv[1])
    if "Target" not in df.columns:
        df["Target"] = 4                        # placeholder so clean() can run
    X, _ = clean(df)
    X = add_features(X).reindex(columns=bundle["columns"], fill_value=0)
    out = pd.DataFrame({"predicted_class": bundle["model"].predict(X)})
    out["label"] = out["predicted_class"].map(TARGET_NAMES)
    print(out.to_string(index=False))
