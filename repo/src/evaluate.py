"""Metrics and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.tree import DecisionTreeClassifier, plot_tree

from .data import TARGET_NAMES

CLASS_LABELS = [TARGET_NAMES[i] for i in (1, 2, 3, 4)]


def score(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, pred, average="weighted")), 4),
    }


def report_text(model, X_test, y_test) -> str:
    return classification_report(y_test, model.predict(X_test), target_names=CLASS_LABELS, digits=3)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(y: pd.Series, out: Path) -> None:
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CLASS_LABELS, counts.values, color=["#b2182b", "#ef8a62", "#fddbc7", "#2166ac"])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / counts.sum():.0%})", ha="center", va="bottom")
    ax.set_ylabel("Households"); ax.set_title("Poverty class distribution")
    ax.set_ylim(0, counts.max() * 1.18); plt.setp(ax.get_xticklabels(), rotation=15)
    _save(fig, out)


def plot_confusion(model, X_test, y_test, out: Path, title: str) -> None:
    cm = confusion_matrix(y_test, model.predict(X_test), labels=[1, 2, 3, 4])
    fig, ax = plt.subplots(figsize=(6.2, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS,
                yticklabels=CLASS_LABELS, ax=ax, cbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=20)
    _save(fig, out)


def plot_importance(tree: DecisionTreeClassifier, columns, out: Path, top: int = 12) -> pd.Series:
    imp = pd.Series(tree.feature_importances_, index=columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    imp.head(top)[::-1].plot.barh(ax=ax, color="#2166ac")
    ax.set_xlabel("Importance (Gini / entropy reduction)"); ax.set_title(f"Top {top} features")
    _save(fig, out)
    return imp


def plot_tree_top(tree: DecisionTreeClassifier, columns, out: Path, depth: int = 3) -> None:
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(tree, max_depth=depth, feature_names=list(columns), class_names=CLASS_LABELS,
              filled=True, rounded=True, fontsize=8, ax=ax, impurity=False, proportion=True)
    ax.set_title(f"Top {depth} levels of the tuned decision tree")
    _save(fig, out)


def plot_depth_curve(X_train, y_train, X_test, y_test, out: Path, random_state: int = 42) -> pd.DataFrame:
    rows = []
    for d in range(1, 21):
        t = DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=random_state).fit(X_train, y_train)
        rows.append((d, t.score(X_train, y_train), t.score(X_test, y_test)))
    df = pd.DataFrame(rows, columns=["max_depth", "train_acc", "test_acc"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df.max_depth, df.train_acc, "o-", label="Train")
    ax.plot(df.max_depth, df.test_acc, "s-", label="Test")
    ax.set_xlabel("max_depth"); ax.set_ylabel("Accuracy"); ax.legend()
    ax.set_title("Overfitting: accuracy vs. tree depth")
    _save(fig, out)
    return df


def plot_model_comparison(results: dict, out: Path) -> None:
    names = list(results)
    f1 = [results[n]["macro_f1"] for n in names]
    acc = [results[n]["accuracy"] for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - w / 2, acc, w, label="Accuracy", color="#67a9cf")
    ax.bar(x + w / 2, f1, w, label="Macro F1", color="#2166ac")
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - w / 2, a + .01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, f + .01, f"{f:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Model comparison on the held-out test set")
    _save(fig, out)
