"""Model definitions: baseline tree, tuned (pruned) tree and comparison models."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42

PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "min_samples_leaf": [1, 5, 10, 20],
    "ccp_alpha": [0.0, 0.0005, 0.001, 0.002],
}


def baseline_tree() -> DecisionTreeClassifier:
    """Unrestricted tree: it grows until every leaf is pure and therefore overfits."""
    return DecisionTreeClassifier(random_state=SEED)


def tuned_tree(cv_splits: int = 5) -> GridSearchCV:
    """Decision tree with class weights + grid search over depth / leaf size / pruning."""
    return GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        PARAM_GRID,
        scoring="f1_macro",
        cv=StratifiedKFold(cv_splits, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )


def comparison_models() -> dict:
    return {
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest (reference)": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1),
    }
