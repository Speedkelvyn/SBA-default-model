# Copyright (c) 2026 Kelvin Afful. MIT License, see LICENSE.
"""Version 2: yearly-refit model. `python run_refit.py data` writes results/refit_*.csv and results/facts_refit.json."""

import json
import sys
import time

import numpy as np
import pandas as pd

import sba_pd as S

OUT = "results"
TEST_YEARS = range(2016, 2022)


def save(df: pd.DataFrame, name: str):
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    return df


def main(folder: str = "data"):
    t0 = time.time()
    d = S.label(S.load(folder))
    d = d[d["ApprovalFY"] >= S.REFIT_FIRST].reset_index(drop=True)
    variants = {
        "A_same_fields": S.features_refit(d, place=False, pricing=False),
        "B_plus_place": S.features_refit(d, place=True, pricing=False),
        "C_plus_place_and_rate": S.features_refit(d, place=True, pricing=True),
    }
    preds = {k: S.refit_forecast(d, x, TEST_YEARS) for k, x in variants.items()}
    facts = {"loans_modeled": int(len(d))}
    rows, pooled = [], []
    for name, p in preds.items():
        y = d.loc[p.index, "default"].to_numpy()
        for v, idx in p.groupby("vintage").indices.items():
            rows.append(
                {
                    "variant": name,
                    "vintage": v,
                    "train_to": v - S.LENDER_LAG,
                    "loans": len(idx),
                    "default_rate": y[idx].mean(),
                    "auc": S.auc(y[idx], p["pd"].to_numpy()[idx]),
                    "ks": S.ks(y[idx], p["pd"].to_numpy()[idx]),
                    "mean_pd": p["pd"].to_numpy()[idx].mean(),
                }
            )
        pooled.append(
            {
                "variant": name,
                "loans": len(p),
                "defaults": int(y.sum()),
                "auc": S.auc(y, p["pd"]),
                "ks": S.ks(y, p["pd"]),
                "mean_pd": p["pd"].mean(),
                "actual": y.mean(),
                "brier": S.brier(y, p["pd"]),
                "brier_constant": y.mean() * (1 - y.mean()),
            }
        )
    save(pd.DataFrame(rows), "refit_by_vintage")
    pool = save(pd.DataFrame(pooled), "refit_pooled")
    print(pool.round(4).to_string(), flush=True)

    pc = preds["C_plus_place_and_rate"]
    pa = preds["A_same_fields"].loc[pc.index]
    dc = d.loc[pc.index]
    y = dc["default"].to_numpy()
    facts["size_only_auc"] = S.auc(y, -np.log(dc["GrossApproval"].clip(lower=1)))
    rng = np.random.default_rng(11)
    boot = []
    for _ in range(100):
        i = rng.integers(0, len(y), len(y))
        boot.append((S.auc(y[i], pc["pd"].to_numpy()[i]), S.auc(y[i], pa["pd"].to_numpy()[i])))
    boot = np.array(boot)
    gap = boot[:, 0] - boot[:, 1]
    facts["boot_C_auc_ci"] = [float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))]
    facts["boot_gap_C_minus_A_ci"] = [float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))]

    # calibration with no recalibration step, and decile spread
    cal = S.calibration(y, pc["pd"].to_numpy())
    save(cal, "refit_calibration")
    facts["decile_low_pred"], facts["decile_high_pred"] = float(cal.predicted.iloc[0]), float(cal.predicted.iloc[-1])
    facts["decile_low_actual"], facts["decile_high_actual"] = float(cal.actual.iloc[0]), float(cal.actual.iloc[-1])

    # dollars: gross charge-offs and the unguaranteed part (loss x (1 - guaranteed share)); recoveries are not in the file
    t = pd.DataFrame(
        {
            "p": pc["pd"].to_numpy(),
            "amt": dc["GrossApproval"].to_numpy(),
            "def": y,
            "gross": dc["loss"].to_numpy(),
            "ung": dc["loss"].to_numpy() * (1 - (dc["SBAGuaranteedApproval"] / dc["GrossApproval"]).clip(0, 1).to_numpy()),
        }
    ).sort_values("p", kind="stable")
    facts["gross_loss_$m"] = float(t["gross"].sum() / 1e6)
    facts["unguaranteed_loss_$m"] = float(t["ung"].sum() / 1e6)
    facts["unguaranteed_share"] = float(t["ung"].sum() / t["gross"].sum())
    facts["approved_$m"] = float(t["amt"].sum() / 1e6)
    cut = []
    for s in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        a = t.iloc[: int(round(len(t) * s))]
        cut.append(
            {
                "approve_share": s,
                "share_of_dollars": a["amt"].sum() / t["amt"].sum(),
                "default_rate": a["def"].mean(),
                "defaults_avoided": 1 - a["def"].sum() / t["def"].sum(),
                "gross_loss_avoided": 1 - a["gross"].sum() / t["gross"].sum(),
                "unguaranteed_loss_avoided": 1 - a["ung"].sum() / t["ung"].sum(),
            }
        )
    save(pd.DataFrame(cut), "refit_cutoffs")

    # audit of the added fields (approvals FY2010 onward, all outcomes)
    m = (d["ChargeOffDate"] - d["ApprovalDate"]).dt.days / 30.44
    dd = d[d["default"] == 1].assign(m=m)
    var = d[d["FixedorVariableInterestInd"] == "V"]
    facts["rate_missing_defaulted"] = float(d.loc[d["default"] == 1, "InitialInterestRate"].isna().mean())
    facts["rate_missing_other"] = float(d.loc[d["default"] == 0, "InitialInterestRate"].isna().mean())
    facts["rate_mean_early_chargeoff"] = float(dd.loc[dd["m"] <= 12, "InitialInterestRate"].mean())
    facts["rate_mean_late_chargeoff"] = float(dd.loc[dd["m"] > 48, "InitialInterestRate"].mean())
    facts["rate_mean_var_defaulted_2018"] = float(
        var.loc[(var["ApprovalFY"] == 2018) & (var["default"] == 1), "InitialInterestRate"].mean()
    )
    facts["rate_mean_var_other_2018"] = float(
        var.loc[(var["ApprovalFY"] == 2018) & (var["default"] == 0), "InitialInterestRate"].mean()
    )
    r3 = lambda v: float((np.round(v * 1000) % 10 != 0).mean())  # noqa: E731
    facts["rate_odd_precision_defaulted"] = r3(d.loc[d["default"] == 1, "InitialInterestRate"].dropna())
    facts["rate_odd_precision_other"] = r3(d.loc[d["default"] == 0, "InitialInterestRate"].dropna())

    # replication of an 18-month early-default design (the SBA overwrites the term at charge-off)
    e = S.load(folder)
    e = e[e["LoanStatus"].isin(["PIF", "CHGOFF", "EXEMPT"]) & e["ApprovalDate"].between("2009-10-01", "2024-03-31")].copy()
    e["early"] = ((e["LoanStatus"] == "CHGOFF") & ((e["ChargeOffDate"] - e["ApprovalDate"]).dt.days <= 548)).astype(int)
    e["whole"] = e["TermInMonths"] % 12 == 0
    from sklearn.ensemble import HistGradientBoostingClassifier

    trn, tst = e[e["ApprovalFY"].between(2010, 2019)], e[e["ApprovalDate"] >= "2021-10-01"]
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0).fit(trn[["TermInMonths"]], trn["early"])
    facts["early_test_loans"] = int(len(tst))
    facts["early_test_defaults"] = int(tst["early"].sum())
    facts["early_whole_year_defaulted"] = float(tst.loc[tst["early"] == 1, "whole"].mean())
    facts["early_whole_year_other"] = float(tst.loc[tst["early"] == 0, "whole"].mean())
    facts["early_term_only_auc"] = S.auc(tst["early"], gb.predict_proba(tst[["TermInMonths"]])[:, 1])
    facts["seconds"] = round(time.time() - t0)
    with open(f"{OUT}/facts_refit.json", "w") as f:
        json.dump(facts, f, indent=1, default=float)
    print(json.dumps(facts, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
