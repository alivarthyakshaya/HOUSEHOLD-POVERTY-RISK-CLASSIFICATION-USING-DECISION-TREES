"""Generate a synthetic household survey with the same *kind* of columns as the
Costa Rican Household Poverty Level Prediction data (IDB / Kaggle).

Target classes (same coding as the Kaggle competition):
    1 = extreme poverty, 2 = moderate poverty, 3 = vulnerable, 4 = non-vulnerable

The data is produced from a latent "wealth" score plus noise, so classes overlap
realistically and a model cannot reach 100 % accuracy. It exists so the whole
pipeline can run end-to-end without downloading anything. For real results, place
the Kaggle ``train.csv`` in ``data/`` (see data/README.md) and re-run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_NAMES = {1: "Extreme poverty", 2: "Moderate poverty", 3: "Vulnerable", 4: "Non-vulnerable"}


def generate(n_households: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_households

    wealth = rng.normal(0, 1, n)                                   # latent socio-economic score
    urban = (rng.random(n) < 1 / (1 + np.exp(-(0.6 * wealth + 0.7)))).astype(int)

    hhsize = np.clip(np.round(rng.normal(4.2 - 0.5 * wealth, 1.6)), 1, 12).astype(int)
    children = np.clip(np.round(hhsize * rng.uniform(0.15, 0.55, n) - 0.3 * wealth),
                       0, hhsize - 1).astype(int)
    adults = hhsize - children
    rooms = np.clip(np.round(rng.normal(4.2 + 0.7 * wealth, 1.3)), 1, 9).astype(int)
    bedrooms = np.clip(np.round(rooms * rng.uniform(0.35, 0.7, n)), 1, rooms).astype(int)

    escolari_head = np.clip(rng.normal(7 + 3.2 * wealth, 3.3), 0, 21)      # head's years of schooling
    meaneduc = np.clip(0.7 * escolari_head + rng.normal(1.5, 1.8, n), 0, 20)
    age_head = np.clip(rng.normal(46, 14, n), 18, 90).round()
    female_head = (rng.random(n) < 1 / (1 + np.exp(0.5 * wealth + 0.9))).astype(int)
    dependency = np.round((children + (age_head > 64) * 0.5) / np.maximum(adults, 1), 3)

    def quality(shift: float) -> np.ndarray:                       # 1 = bad, 2 = regular, 3 = good
        s = 0.9 * wealth + shift + rng.normal(0, 0.9, n)
        return np.digitize(s, [-0.4, 0.7]) + 1

    wall_q, floor_q, roof_q = quality(0.1), quality(0.0), quality(0.2)

    def owns(shift: float) -> np.ndarray:
        return (rng.random(n) < 1 / (1 + np.exp(-(0.9 * wealth + shift)))).astype(int)

    refrig, television, computer = owns(1.8), owns(1.2), owns(-0.6)
    has_tablet = owns(-0.9)
    tablets = np.where(has_tablet == 1, rng.integers(1, 4, n), 0)
    cellphones = np.clip(np.round(rng.normal(1.5 + 0.8 * wealth + 0.25 * adults, 0.9)), 0, 8).astype(int)
    bathroom, electricity, water_inside = owns(2.2), owns(3.5), owns(1.6)

    tipo = rng.choice([0, 1, 2, 3], n, p=[0.55, 0.2, 0.1, 0.15])   # 0 owned, 1 rented, 2 precarious, 3 other
    rent = np.where(tipo == 1, np.clip(rng.normal(150000 + 60000 * wealth, 50000), 20000, None), np.nan)
    overcrowding = np.round(hhsize / rooms, 3)

    meaneduc_obs = meaneduc.copy()
    meaneduc_obs[rng.random(n) < 0.02] = np.nan                    # realistic missing values

    score = (wealth
             + 0.25 * (wall_q + floor_q + roof_q - 6) / 3
             + 0.15 * (refrig + television + computer - 1.5)
             - 0.25 * (overcrowding - 1)
             - 0.3 * dependency
             + 0.04 * (meaneduc - 7)
             + rng.normal(0, 0.25, n))
    cut = np.quantile(score, [0.08, 0.25, 0.39])                   # ~8 / 17 / 14 / 61 % like the Kaggle data
    target = np.digitize(score, cut) + 1

    return pd.DataFrame({
        "Id": [f"ID_{i:05d}" for i in range(n)],
        "v2a1": rent, "rooms": rooms, "bedrooms": bedrooms, "v14a": bathroom,
        "refrig": refrig, "v18q": has_tablet, "v18q1": tablets,
        "r4t1": children, "r4t2": adults, "hhsize": hhsize,
        "escolari_head": escolari_head.round(1), "meaneduc": meaneduc_obs.round(2),
        "age_head": age_head, "female_head": female_head, "dependency": dependency,
        "epared": wall_q, "etecho": roof_q, "eviv": floor_q,
        "television": television, "computer": computer, "qmobilephone": cellphones,
        "electricity": electricity, "water_inside": water_inside,
        "tipovivi": tipo, "area_urban": urban, "overcrowding": overcrowding,
        "Target": target,
    })


if __name__ == "__main__":
    out = generate()
    out.to_csv("data/synthetic_households.csv", index=False)
    print(out.shape, out["Target"].value_counts(normalize=True).sort_index().round(3).to_dict())
