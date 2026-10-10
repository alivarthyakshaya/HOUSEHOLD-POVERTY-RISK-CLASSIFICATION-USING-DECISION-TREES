"""Household Poverty Risk Classification using Decision Trees.

Usage
-----
    python main.py                          # uses data/train.csv if present, else synthetic data
    python main.py --data data/train.csv    # explicit Kaggle file
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.tree import export_text

from src import evaluate as ev
from src.data import add_features, clean, load_raw
from src.models import SEED, baseline_tree, comparison_models, tuned_tree


def main(data_path: str | None, out_dir: str = "outputs") -> dict:
    out = Path(out_dir); fig = out / "figures"; fig.mkdir(parents=True, exist_ok=True)

    raw, source = load_raw(data_path)
    X, y = clean(raw)
    X = add_features(X)
    print(f"[data] source={source}  households={len(X)}  features={X.shape[1]}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    ev.plot_class_distribution(y, fig / "class_distribution.png")

    results: dict = {}

    base = baseline_tree().fit(X_tr, y_tr)
    results["Decision Tree (baseline, unpruned)"] = ev.score(base, X_te, y_te)
    results["Decision Tree (baseline, unpruned)"]["depth"] = int(base.get_depth())
    results["Decision Tree (baseline, unpruned)"]["train_accuracy"] = round(float(base.score(X_tr, y_tr)), 4)

    search = tuned_tree().fit(X_tr, y_tr)
    best = search.best_estimator_
    results["Decision Tree (tuned, proposed)"] = ev.score(best, X_te, y_te)
    results["Decision Tree (tuned, proposed)"]["depth"] = int(best.get_depth())
    results["Decision Tree (tuned, proposed)"]["leaves"] = int(best.get_n_leaves())
    results["Decision Tree (tuned, proposed)"]["train_accuracy"] = round(float(best.score(X_tr, y_tr)), 4)
    results["Decision Tree (tuned, proposed)"]["cv_macro_f1"] = round(float(search.best_score_), 4)
    print("[tuning] best params:", search.best_params_)

    for name, model in comparison_models().items():
        model.fit(X_tr, y_tr)
        results[name] = ev.score(model, X_te, y_te)

    for name, r in results.items():
        print(f"  {name:38s} acc={r['accuracy']:.3f}  macroF1={r['macro_f1']:.3f}")

    # stability of the proposed tree across folds on the full data
    cv = cross_val_score(tuned_tree(3).estimator.set_params(**search.best_params_), X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1_macro")

    # plots / artefacts
    ev.plot_confusion(best, X_te, y_te, fig / "confusion_matrix_tuned.png", "Tuned decision tree")
    ev.plot_confusion(base, X_te, y_te, fig / "confusion_matrix_baseline.png", "Baseline (unpruned) tree")
    imp = ev.plot_importance(best, X.columns, fig / "feature_importance.png")
    ev.plot_tree_top(best, X.columns, fig / "decision_tree_top3.png")
    depth_df = ev.plot_depth_curve(X_tr, y_tr, X_te, y_te, fig / "depth_vs_accuracy.png")
    ev.plot_model_comparison(results, fig / "model_comparison.png")

    (out / "classification_report_tuned.txt").write_text(ev.report_text(best, X_te, y_te))
    (out / "tree_rules.txt").write_text(export_text(best, feature_names=list(X.columns), max_depth=4))
    imp.head(15).to_csv(out / "feature_importance.csv", header=["importance"])
    depth_df.to_csv(out / "depth_curve.csv", index=False)
    joblib.dump({"model": best, "columns": list(X.columns)}, out / "decision_tree_model.joblib")

    summary = {"data_source": source, "n_households": int(len(X)), "n_features": int(X.shape[1]),
               "train_size": int(len(X_tr)), "test_size": int(len(X_te)),
               "class_counts": {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
               "best_params": search.best_params_,
               "cv5_macro_f1_mean": round(float(cv.mean()), 4), "cv5_macro_f1_std": round(float(cv.std()), 4),
               "results": results, "top_features": {k: round(float(v), 4) for k, v in imp.head(10).items()}}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] artefacts written to {out}/")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv", help="path to Kaggle train.csv (optional)")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    main(a.data, a.out)
