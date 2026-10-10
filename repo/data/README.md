# Data

The project uses the **Costa Rican Household Poverty Level Prediction** dataset
(Inter-American Development Bank, hosted on Kaggle).

1. Download `train.csv` from <https://www.kaggle.com/competitions/costa-rican-household-poverty-prediction/data>
   (a free Kaggle account is required; accept the competition rules).
2. Save it as `data/train.csv`.
3. Run `python main.py`.

If `data/train.csv` is not found, `main.py` automatically generates a **synthetic** dataset with a
similar structure (`src/generate_synthetic_data.py`, also saved as `synthetic_households.csv`) so the
pipeline can be run and tested anywhere. Results on synthetic data are for demonstration only and are
not real poverty statistics.

Target coding: 1 = extreme poverty, 2 = moderate poverty, 3 = vulnerable, 4 = non-vulnerable.
