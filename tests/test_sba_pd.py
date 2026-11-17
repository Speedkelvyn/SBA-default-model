# Copyright (c) 2026 Kelvin Afful. MIT License, see LICENSE.
"""Tests for sba_pd.py: labelling rules, the lender record, metrics and the dollar table, on small hand-built data."""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import sba_pd as S  # noqa: E402


def loans(rows):
    base = {
        "GrossApproval": 100_000.0,
        "SBAGuaranteedApproval": 75_000.0,
        "ApprovalFY": 2008,
        "LoanStatus": "PIF",
        "ChargeOffDate": pd.NaT,
        "GrossChargeOffAmount": 0.0,
        "TermInMonths": 84,
        "LocationID": "A",
        "ApprovalDate": pd.Timestamp("2008-01-15"),
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_label_horizon_and_window():
    d = loans(
        [
            {"LoanStatus": "CHGOFF", "ChargeOffDate": pd.Timestamp("2012-12-31"), "GrossChargeOffAmount": 40_000.0},  # 59 months
            {"LoanStatus": "CHGOFF", "ChargeOffDate": pd.Timestamp("2013-03-01"), "GrossChargeOffAmount": 40_000.0},  # 62 months
            {"LoanStatus": "PIF"},
            {"LoanStatus": "EXEMPT"},
            {"LoanStatus": "CANCLD"},  # never disbursed: not a loan outcome
            {"LoanStatus": "CHGOFF", "ChargeOffDate": pd.NaT},  # timing unknown: dropped
            {"ApprovalDate": pd.Timestamp("2023-01-15"), "ApprovalFY": 2023},  # window not closed yet: dropped
            {"GrossApproval": -5.0},
        ]
    )
    out = S.label(d)
    assert out["default"].tolist() == [1, 0, 0, 0]
    assert out["loss"].tolist() == [40_000.0, 0.0, 0.0, 0.0]  # a charge-off after the horizon is not a loss in the window


def test_features_leave_out_the_outcome_fields():
    d = S.label(
        loans(
            [
                {
                    "LoanStatus": "PIF",
                    "BusinessAge": "Existing, 5 or more years",
                    "ProcessingMethod": "7a General",
                    "NaicsCode": "722511",
                    "BusinessType": "CORPORATION",
                    "RevolverStatus": "N",
                    "CollateralInd": "Y",
                    "FranchiseCode": None,
                    "BorrState": "TX",
                    "JobsSupported": 4,
                }
            ]
        )
    )
    x = S.features(d)
    for f in S.OUTCOME_FIELDS:
        assert f not in x.columns
    assert x.loc[0, "sector"] == "72" and x.loc[0, "age_group"] == "established" and x.loc[0, "franchise"] == "N"
    assert x.loc[0, "guarantee_share"] == pytest.approx(0.75) and x.loc[0, "jobs_per_100k"] == pytest.approx(4.0)


def test_lender_record_uses_only_closed_windows():
    rows = []
    for fy, n, bad in ((2000, 10, 5), (2003, 10, 0), (2005, 10, 10)):
        for i in range(n):
            rows.append(
                {"ApprovalFY": fy, "LoanStatus": "CHGOFF" if i < bad else "PIF", "ApprovalDate": pd.Timestamp(f"{fy}-03-01")}
            )
    d = loans(rows)
    d["default"] = (d["LoanStatus"] == "CHGOFF").astype(int)
    probe = loans([{"ApprovalFY": 2006}, {"ApprovalFY": 2009}, {"ApprovalFY": 2011}, {"LocationID": "B", "ApprovalFY": 2011}])
    probe["default"] = 0
    both = pd.concat([d, probe], ignore_index=True)
    h = S.lender_history(both, lag=6, k=20)
    i2006, i2009, i2011, ib = h.index[-4:]
    # FY2006 loan sees FY<=2000 only: 5 defaults of 10, pooled rate .5 -> (5 + 20*.5)/(10+20)
    assert h.loc[i2006, "lender_rate"] == pytest.approx(0.5)
    assert h.loc[i2006, "lender_log_loans"] == pytest.approx(math.log1p(10))
    # FY2009 loan sees FY<=2003: 5 defaults of 20; pooled the same
    assert h.loc[i2009, "lender_rate"] == pytest.approx((5 + 20 * 0.25) / (20 + 20))
    # FY2011 loan sees FY<=2005, so the FY2005 loans, whose windows close in 2010, are in
    assert h.loc[i2011, "lender_log_loans"] == pytest.approx(math.log1p(30))
    # a lender with no history gets the pooled rate and no loans
    assert h.loc[ib, "lender_log_loans"] == 0 and h.loc[ib, "lender_rate"] == pytest.approx(15 / 30)


def test_auc_ks_brier_hand_computed():
    y = np.array([0, 0, 1, 1])
    assert S.auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0 and S.auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert S.auc(y, [0.5] * 4) == 0.5 and math.isnan(S.auc(np.zeros(4), np.arange(4.0)))
    assert S.auc([0, 1, 0, 1], [0.1, 0.35, 0.4, 0.8]) == pytest.approx(0.75)
    assert S.ks(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert S.ks([0, 1, 0, 1], [0.1, 0.35, 0.4, 0.8]) == pytest.approx(0.5)
    assert S.brier([1, 0], [0.8, 0.3]) == pytest.approx((0.04 + 0.09) / 2)


def test_psi_zero_for_same_and_positive_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20_000)
    assert S.psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert S.psi(a, a + 1.0) > 0.25


def test_shift_intercept_hits_target_mean():
    p = np.random.default_rng(1).beta(2, 8, size=5000)
    q = S.shift_intercept(p, 0.05)
    assert q.mean() == pytest.approx(0.05, abs=1e-6)
    assert np.all(np.argsort(p) == np.argsort(q))  # ranking unchanged


def test_scorecard_learns_a_clear_pattern_and_scores_unseen_levels():
    rng = np.random.default_rng(2)
    n = 6000
    risky = rng.random(n) < 0.5
    x = pd.DataFrame({c: rng.normal(size=n) for c in S.NUMERIC})
    for c in S.CATEGORICAL:
        x[c] = "a"
    x["method"] = np.where(risky, "low_doc", "preferred_lender")
    y = (rng.random(n) < np.where(risky, 0.3, 0.03)).astype(int)
    sc = S.Scorecard().fit(x, y)
    p = sc.predict(x)
    assert S.auc(y, p) > 0.7 and p[risky].mean() > 2 * p[~risky].mean()
    x2 = x.head(5).copy()
    x2["method"] = "never_seen"
    assert np.isfinite(sc.predict(x2)).all()
    pts = sc.points()
    low = pts[(pts.variable == "method") & (pts.level == "low_doc")]["points"].iloc[0]
    pref = pts[(pts.variable == "method") & (pts.level == "preferred_lender")]["points"].iloc[0]
    assert low < pref  # riskier level, fewer points


def test_cutoff_table_arithmetic():
    d = pd.DataFrame({"GrossApproval": [100.0, 100.0, 100.0, 100.0], "default": [0, 0, 1, 1], "loss": [0, 0, 60.0, 40.0]})
    p = [0.1, 0.2, 0.3, 0.9]  # the riskiest loan is a default
    t = S.cutoff_table(d, p, shares=(1.0, 0.75, 0.5)).set_index("approve_share")
    assert t.loc[1.0, "default_rate"] == 0.5 and t.loc[1.0, "loss_avoided_share"] == 0
    assert t.loc[0.75, "loans"] == 3 and t.loc[0.75, "defaults_avoided"] == pytest.approx(0.5)
    assert t.loc[0.75, "loss_avoided_share"] == pytest.approx(0.4)
    assert t.loc[0.5, "default_rate"] == 0 and t.loc[0.5, "loss_avoided_share"] == pytest.approx(1.0)
    assert S.expected_loss(d, p, 0.5) == pytest.approx(0.5 * 100 * sum(p))


def test_calibration_bins_match_hand_values():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    p = np.array([0.05, 0.1, 0.15, 0.2, 0.6, 0.7, 0.8, 0.9])
    t = S.calibration(y, p, bins=2)
    assert t["predicted"].tolist() == pytest.approx([0.125, 0.75]) and t["actual"].tolist() == [0.25, 0.75]


def _refit_frame(n_per_year=1200, years=range(2010, 2018)):
    rng = np.random.default_rng(1)
    rows = []
    for fy in years:
        for i in range(n_per_year):
            size = float(rng.choice([30_000.0, 150_000.0, 600_000.0]))
            bad = int(rng.random() < (0.08 if size < 100_000 else 0.02))
            rows.append(
                {
                    "ApprovalFY": fy,
                    "GrossApproval": size,
                    "SBAGuaranteedApproval": 0.75 * size,
                    "default": bad,
                    "BankState": "TX" if i % 3 else "CA",
                    "BorrState": "TX",
                    "ProjectState": "TX",
                    "InitialInterestRate": 6.0 + 2.0 * bad + rng.normal(0, 0.3),
                    "FixedorVariableInterestInd": "V" if i % 2 else "F",
                    "NaicsCode": "722511",
                    "BusinessType": "CORPORATION",
                    "BusinessAge": "Existing, 5 or more years",
                    "ProcessingMethod": "7a General",
                    "RevolverStatus": "N",
                    "CollateralInd": "Y",
                    "FranchiseCode": None,
                    "JobsSupported": 3.0,
                    "ApprovalDate": pd.Timestamp(f"{fy}-03-01"),
                    "LocationID": "A",
                }
            )
    return pd.DataFrame(rows)


def test_features_refit_adds_place_and_pricing_and_no_outcomes():
    d = _refit_frame()
    x = S.features_refit(d)
    for c in S.PLACE + S.PRICING:
        assert c in x.columns
    assert not set(S.LENDER) & set(x.columns)
    assert not {"default", "TermInMonths", "LoanStatus", "GrossChargeOffAmount"} & set(x.columns)
    assert x["out_of_state"].tolist()[:3] == [1.0, 0.0, 0.0] or set(x["out_of_state"].unique()) <= {0.0, 1.0}
    assert S.features_refit(d, place=False, pricing=False).shape[1] == S.features_refit(d).shape[1] - 4


def test_refit_forecast_trains_only_on_loans_old_enough_and_ranks():
    d = _refit_frame()
    x = S.features_refit(d)
    out = S.refit_forecast(d, x, test_years=[2016, 2017], first=2010, lag=6)
    assert set(out["vintage"]) == {2016, 2017}
    assert (d.loc[out.index, "ApprovalFY"] == out["vintage"]).all()
    # a 2016 loan is scored by a model fitted on FY2010 only; a 2017 loan on FY2010 and FY2011
    assert S.auc(d.loc[out.index, "default"], out["pd"]) > 0.6  # the planted size and rate signal is found
