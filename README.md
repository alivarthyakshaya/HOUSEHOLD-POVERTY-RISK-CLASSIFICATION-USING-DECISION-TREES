# Household Poverty Risk Classification Using Decision Trees

Course End Project - Machine Learning (A8703), B.Tech CSE, Section G (3rd year), Vardhaman College of Engineering, 2026-27.

**Team:** ACHA BHUVANAKRUTHI (24881A05BZ) · ADLA NITHIN (24881A05CA) · ALIVARTHY AKSHAYA (24881A05CB)

## What it does
Predicts a household's poverty level from housing quality, assets, education and family structure, using
a **Decision Tree** classifier (four classes: extreme poverty, moderate poverty, vulnerable, non-vulnerable).
Decision trees were chosen because their IF-THEN rules can be read and audited by social-welfare analysts.

## Pipeline
`load -> clean (head-of-household rows, yes/no -> 1/0, impute) -> feature engineering -> 80/20 stratified split
-> baseline tree -> tuned tree (GridSearchCV, 5-fold, macro-F1, class weights, cost-complexity pruning)
-> compare with Logistic Regression / Random Forest -> plots + metrics`

## Project layout
```
main.py                      # run the whole pipeline
predict.py                   # score a new CSV with the saved model
src/data.py                  # loading, cleaning, feature engineering
src/models.py                # baseline tree, tuned tree, comparison models
src/evaluate.py              # metrics and plots
src/generate_synthetic_data.py
tests/test_pipeline.py       # pytest
data/README.md               # how to get the real dataset
outputs/                     # metrics.json, tree_rules.txt, figures/, saved model
```

## Quick start
```bash
git clone https://github.com/<your-username>/household-poverty-decision-tree.git
cd household-poverty-decision-tree
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                      # real data if data/train.csv exists, else synthetic
pytest -q                           # run tests
```

## Results (synthetic demo data, 3000 households, 20 % test split)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority-class baseline | 0.610 | 0.189 |
| Decision Tree, unpruned (depth 17, train acc 1.00) | 0.723 | 0.596 |
| **Decision Tree, tuned (depth 8, entropy)** | 0.682 | 0.566 |
| Logistic Regression | 0.760 | 0.679 |
| Random Forest (reference) | 0.785 | 0.637 |

Numbers come from synthetic data and will change on the real Kaggle file; re-run `python main.py` and read `outputs/metrics.json`.
Top features: head's years of schooling, wall quality, asset score, roof quality, overcrowding.

## License
MIT
