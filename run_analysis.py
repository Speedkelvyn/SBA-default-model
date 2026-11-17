# Copyright (c) 2026 Kelvin Afful. MIT License, see LICENSE.
"""Reproduce every number in MODEL_DOCUMENT.md: `python run_analysis.py data` (folder holding the slim SBA files)."""

import json
import sys
import time

import numpy as np
import pandas as pd

import sba_pd as S

OUT = "results"


def save(df: pd.DataFrame, name: str, index=False):
    df.to_csv(f"{OUT}/{name}.csv", index=index)
    return df


def main(folder: str = "data"):
    t0 = time.time()
    raw = S.load(folder)
    d = S.label(raw)
    xall = S.features(d)
    cols = S.NUMERIC + S.CATEGORICAL
    x = xall[cols]
    facts = {"loans_in_file": int(len(raw)), "statuses": raw["LoanStatus"].value_counts().to_dict()}
    facts["population"] = int(len(d))
    vint = d.groupby("ApprovalFY").agg(
        loans=("default", "size"), default_rate=("default", "mean"), approved_m=("GrossApproval", lambda s: s.sum() / 1e6)
    )
    save(vint.reset_index(), "vintages")
    tr_mask = d["ApprovalFY"].between(*S.TRAIN_FY)
    te_mask = d["ApprovalFY"].between(*S.TEST_FY)
    st_mask = d["ApprovalFY"].between(*S.STRICT_TEST_FY)
    tr, te, st = d[tr_mask], d[te_mask], d[st_mask]
    facts.update(
        train_loans=len(tr),
        test_loans=len(te),
        strict_loans=len(st),
        train_rate=tr["default"].mean(),
        test_rate=te["default"].mean(),
        strict_rate=st["default"].mean(),
    )

    # --- fixed split: train FY2001-2009, test FY2010-2021
    sc, gbm, ((s_tr, g_tr), (s_te, g_te)) = S.fit_predict(tr, x[tr_mask], [x[tr_mask], x[te_mask]], gbm_iter=400)
    ytr, yte = tr["default"].to_numpy(), te["default"].to_numpy()
    in_strict = (te["ApprovalFY"] >= S.STRICT_TEST_FY[0]).to_numpy()
    rows = []
    for name, (ptr, pte) in {"scorecard": (s_tr, s_te), "boosted": (g_tr, g_te)}.items():
        rows.append(
            {
                "model": name,
                "train_auc": S.auc(ytr, ptr),
                "test_auc": S.auc(yte, pte),
                "strict_test_auc": S.auc(yte[in_strict], pte[in_strict]),
                "train_ks": S.ks(ytr, ptr),
                "test_ks": S.ks(yte, pte),
                "strict_test_ks": S.ks(yte[in_strict], pte[in_strict]),
                "test_brier": S.brier(yte, pte),
                "mean_pd_test": pte.mean(),
                "actual_test": yte.mean(),
            }
        )
    perf = save(pd.DataFrame(rows), "performance")
    print(perf.round(3).to_string(), flush=True)

    # --- by vintage
    bv = []
    for y, idx in te.groupby("ApprovalFY").indices.items():
        bv.append(
            {
                "vintage": y,
                "loans": len(idx),
                "default_rate": yte[idx].mean(),
                "scorecard_auc": S.auc(yte[idx], s_te[idx]),
                "boosted_auc": S.auc(yte[idx], g_te[idx]),
                "mean_pd_boosted": g_te[idx].mean(),
                "psi_boosted_vs_train": S.psi(g_tr, g_te[idx]),
            }
        )
    save(pd.DataFrame(bv), "by_vintage")

    # --- placebo: labels shuffled within vintage
    rng = np.random.default_rng(0)
    pl = []
    for _ in range(5):
        shuffled = te.groupby("ApprovalFY")["default"].transform(lambda s: rng.permutation(s.to_numpy())).to_numpy()
        pl.append(S.auc(shuffled, g_te))
    facts["placebo_auc_mean"], facts["placebo_auc_sd"] = float(np.mean(pl)), float(np.std(pl))

    # --- calibration and level recalibration (real time: shift on FY2010-2014, apply to FY2015-2021)
    save(S.calibration(yte, g_te), "calibration_boosted")
    save(S.calibration(yte, s_te), "calibration_scorecard")
    early = (te["ApprovalFY"] <= 2014).to_numpy()
    g_rc_early = S.shift_intercept(g_te[early], yte[early].mean())
    shift = float(np.mean(S.logit(g_rc_early) - S.logit(g_te[early])))  # one constant on the log-odds scale
    g_rc = 1 / (1 + np.exp(-(S.logit(g_te) + shift)))
    facts["recal_shift_logodds"] = shift
    rec = pd.DataFrame(
        [
            {
                "scope": "FY2015-2021",
                "mean_pd_raw": g_te[~early].mean(),
                "mean_pd_recalibrated": g_rc[~early].mean(),
                "actual": yte[~early].mean(),
                "brier_raw": S.brier(yte[~early], g_te[~early]),
                "brier_recalibrated": S.brier(yte[~early], g_rc[~early]),
            }
        ]
    )
    save(rec, "recalibration")

    # --- drivers
    save(sc.points().round(3), "scorecard_points")
    sub = te.sample(min(80_000, len(te)), random_state=3)
    xs = S.boosted_frame(x.loc[sub.index], x[tr_mask])
    base = S.auc(sub["default"], gbm.predict_proba(xs)[:, 1])
    imp = []
    for c in cols:
        xp = xs.copy()
        xp[c] = xp[c].sample(frac=1, random_state=5).to_numpy()
        imp.append({"feature": c, "auc_drop": base - S.auc(sub["default"], gbm.predict_proba(xp)[:, 1])})
    save(pd.DataFrame(imp).sort_values("auc_drop", ascending=False), "drivers_boosted")

    # --- simple benchmarks and sampling error (real-time test years)
    ys = yte[in_strict]
    facts["strict_defaults"] = int(ys.sum())
    facts["auc_size_only_strict"] = S.auc(ys, -x.loc[te.index[in_strict], "log_amount"].to_numpy())
    facts["brier_constant_strict"] = float(ys.mean() * (1 - ys.mean()))
    rng2 = np.random.default_rng(11)
    boot = []
    for _ in range(100):
        i = rng2.integers(0, len(ys), len(ys))
        boot.append((S.auc(ys[i], g_te[in_strict][i]), S.auc(ys[i], s_te[in_strict][i])))
    boot = np.array(boot)

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    facts["boot_boosted_auc_ci"] = ci(boot[:, 0])
    facts["boot_scorecard_auc_ci"] = ci(boot[:, 1])
    facts["boot_gap_ci"] = ci(boot[:, 0] - boot[:, 1])

    # --- what carries the signal: scorecard ablations on the same split
    size_auc = S.auc(yte, -x.loc[te.index, "log_amount"].to_numpy())
    abl = [
        {
            "spec": "loan size alone, monotone (smaller loan = riskier)",
            "test_auc": size_auc,
            "strict_test_auc": facts["auc_size_only_strict"],
        }
    ]
    for name, keep in {
        "scorecard, all features": cols,
        "scorecard, no state": [c for c in cols if c != "state"],
        "scorecard, no state or sector": [c for c in cols if c not in ("state", "sector")],
        "scorecard, loan size only (binned)": ["log_amount"],
    }.items():
        m_ = S.Scorecard().fit(x.loc[tr_mask, keep], ytr)
        p_ = m_.predict(x.loc[te_mask, keep])
        abl.append({"spec": name, "test_auc": S.auc(yte, p_), "strict_test_auc": S.auc(yte[in_strict], p_[in_strict])})
    abl.append({"spec": "scorecard, 12 bins per number", "test_auc": np.nan, "strict_test_auc": np.nan})
    m12 = S.Scorecard(bins=12).fit(x[tr_mask], ytr)
    p12 = m12.predict(x[te_mask])
    abl[-1].update(test_auc=S.auc(yte, p12), strict_test_auc=S.auc(yte[in_strict], p12[in_strict]))
    save(pd.DataFrame(abl), "ablation_scorecard")

    # --- the lender record, tested as an addition (not part of the model)
    xl = xall[cols + S.LENDER]
    _, _, ((sl_te, gl_te),) = S.fit_predict(tr, xl[tr_mask], [xl[te_mask]], gbm_iter=400)
    save(
        pd.DataFrame(
            [
                {
                    "model": "scorecard + lender record",
                    "test_auc": S.auc(yte, sl_te),
                    "strict_test_auc": S.auc(yte[in_strict], sl_te[in_strict]),
                },
                {
                    "model": "boosted + lender record",
                    "test_auc": S.auc(yte, gl_te),
                    "strict_test_auc": S.auc(yte[in_strict], gl_te[in_strict]),
                },
            ]
        ),
        "with_lender_record",
    )

    # --- dollars on the real-time test years (FY2015-2021), boosted model, level-recalibrated
    ds = te[in_strict]
    ct = S.cutoff_table(ds, g_te[in_strict])
    save(ct, "cutoffs_boosted")
    save(S.cutoff_table(ds, s_te[in_strict]), "cutoffs_scorecard")
    lgd = float((tr["loss"].sum()) / max(tr.loc[tr["default"] == 1, "GrossApproval"].sum(), 1))
    facts["lgd_train"] = lgd
    facts["expected_loss_boosted_$m"] = S.expected_loss(ds, g_rc[in_strict], lgd) / 1e6
    facts["actual_loss_$m"] = float(ds["loss"].sum() / 1e6)
    facts["approved_$m"] = float(ds["GrossApproval"].sum() / 1e6)

    # --- leak check on the excluded field
    lk = d[d["ApprovalFY"].between(2010, 2016)]
    facts["term_whole_years_defaulted"] = float((lk.loc[lk["default"] == 1, "TermInMonths"] % 12 == 0).mean())
    facts["term_whole_years_other"] = float((lk.loc[lk["default"] == 0, "TermInMonths"] % 12 == 0).mean())

    # --- walk-forward
    wf = S.walk_forward(d, x)
    save(wf, "walk_forward")
    print(wf.round(3).to_string(), flush=True)
    facts["seconds"] = round(time.time() - t0)
    with open(f"{OUT}/facts.json", "w") as f:
        json.dump(facts, f, indent=1, default=float)
    print(json.dumps(facts, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
