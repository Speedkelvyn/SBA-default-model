# Copyright (c) 2026 Kelvin Afful. MIT License, see LICENSE.
"""Audit of the 18-month early-default design for other leaked fields. `python run_early_audit.py data` writes
results/early_audit.csv and results/facts_early.json (about 1.5 minutes)."""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import sba_pd as S

OUT = "results"
EARLY_DAYS = 548  # 18 months


def fit_scores(x, y, trn, tst, cols):
    xs = x[cols].copy()
    for c in cols:
        if xs[c].dtype == object:
            xs[c] = xs[c].astype("category")
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0, categorical_features="from_dtype")
    m.fit(xs[trn], y[trn])
    return m.predict_proba(xs[tst])[:, 1]


def main(folder: str = "data"):
    e = S.load(folder)
    e = e[e["LoanStatus"].isin(["PIF", "CHGOFF", "EXEMPT"]) & e["ApprovalDate"].between("2009-10-01", "2024-03-31")].copy()
    e["early"] = ((e["LoanStatus"] == "CHGOFF") & ((e["ChargeOffDate"] - e["ApprovalDate"]).dt.days <= EARLY_DAYS)).astype(int)
    e["default"] = e["early"]  # features() builds a lender record from `default`; it is dropped below
    x = S.features_refit(e)
    x["term_months"] = e["TermInMonths"].astype(float)
    y = e["early"].to_numpy()
    trn = e["ApprovalFY"].between(2010, 2019).to_numpy()
    tst = (e["ApprovalDate"] >= "2021-10-01").to_numpy()
    cols = list(x.columns)
    base = [c for c in cols if c != "term_months"]
    yt = y[tst]
    rows = []
    p_all = fit_scores(x, y, trn, tst, cols)
    p_base = fit_scores(x, y, trn, tst, base)
    for c in cols:
        rows.append(
            {
                "field": c,
                "alone_auc": S.auc(yt, fit_scores(x, y, trn, tst, [c])),
                "auc_without_term_and_without_field": np.nan
                if c == "term_months"
                else S.auc(yt, fit_scores(x, y, trn, tst, [k for k in base if k != c])),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/early_audit.csv", index=False)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(300):
        i = rng.integers(0, len(yt), len(yt))
        boots.append(S.auc(yt[i], p_base[i]))
    d_te, d_tr = e[tst], e[trn]
    facts = {
        "test_loans": int(tst.sum()),
        "test_defaults": int(yt.sum()),
        "train_defaults": int(y[trn].sum()),
        "auc_all": S.auc(yt, p_all),
        "auc_without_term": S.auc(yt, p_base),
        "auc_without_term_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "median_amount_early": float(d_te.loc[d_te["early"] == 1, "GrossApproval"].median()),
        "median_amount_other": float(d_te.loc[d_te["early"] == 0, "GrossApproval"].median()),
        "median_amount_early_train": float(d_tr.loc[d_tr["early"] == 1, "GrossApproval"].median()),
        "median_amount_other_train": float(d_tr.loc[d_tr["early"] == 0, "GrossApproval"].median()),
        "size_alone_auc": float(df.loc[df["field"] == "log_amount", "alone_auc"].iloc[0]),
        "size_drop": float(S.auc(yt, p_base) - df.loc[df["field"] == "log_amount", "auc_without_term_and_without_field"].iloc[0]),
        "max_other_drop": float(
            (
                S.auc(yt, p_base) - df.loc[~df["field"].isin(["term_months", "log_amount"]), "auc_without_term_and_without_field"]
            ).max()
        ),
    }
    with open(f"{OUT}/facts_early.json", "w") as f:
        json.dump(facts, f, indent=1, default=float)
    print(json.dumps(facts, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
