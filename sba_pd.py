# Copyright (c) 2026 Kelvin Afful. MIT License, see LICENSE.
"""
sba_pd.py -- probability-of-default model for SBA 7(a) small-business loans, built on the SBA's public FOIA loan file.

Question answered: at approval, from what the lender knows about the loan and the borrower, how likely is the loan to be
charged off within 60 months, and does ranking loans by that likelihood hold up on loans approved after the model was built?

    label       charged off within 60 months of approval (loans paid in full or still active count as not defaulted)
    population  loans approved at least 60 months before the file date, so every loan has a full window
    features    only what is known at approval; nothing from the outcome fields
    split       train on FY2001 to FY2009 approvals, test on later approvals (out of time)
    models      a binned logistic scorecard (the bank baseline) and gradient boosting

Run `python run_analysis.py data` to reproduce results/.
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-30")  # the file date of the SBA release
HORIZON = 60  # months
TRAIN_FY = (2001, 2009)  # NAICS industry codes exist from FY2001; interest rates only from FY2009, so rates are not used
TEST_FY = (2010, 2021)
STRICT_TEST_FY = (2015, 2021)  # approvals made after the last training loan's 60-month window had closed (real-time test)
LENDER_LAG = 6  # a lender's record for a FY-v loan uses FY <= v-6 loans, whose 60-month windows closed before it was approved
OUTCOME_FIELDS = ["LoanStatus", "PaidInFullDate", "ChargeOffDate", "GrossChargeOffAmount", "SoldSecMrktInd", "TermInMonths"]

AGE_GROUP = {
    "Startup, Loan Funds will Open Business": "new",
    "New, Less than 1 Year old": "new",
    "New Business or 2 years or less": "new",
    "Less than 3 years old but at least 2": "young",
    "Less than 4 years old but at least 3": "young",
    "Less than 5 years old but at least 4": "young",
    "Existing, 5 or more years": "established",
    "Existing or more than 2 years old": "established",
    "Change of Ownership": "ownership_change",
}
METHOD_GROUP = {
    "SBA Express Program": "express",
    "Preferred Lenders Program": "preferred_lender",
    "7a General": "general",
    "Low Documentation Program": "low_doc",
    "Certified Lenders Program": "certified_lender",
}
NUMERIC = ["log_amount", "guarantee_share", "jobs_per_100k"]
LENDER = ["lender_rate", "lender_log_loans"]  # tested as an addition and rejected (see MODEL_DOCUMENT.md)
CATEGORICAL = ["sector", "business_type", "age_group", "method", "revolver", "collateral", "franchise", "state"]


# ------------------------------------------------------------------ data
def load(folder: str) -> pd.DataFrame:
    """Read the slim files written by slim_sba.py (paid in full, charged off, and still-active loans)."""
    files = sorted(glob.glob(os.path.join(folder, "SBA_slim_part*.csv.gz"))) + sorted(
        glob.glob(os.path.join(folder, "SBA_exempt_part*.csv.gz"))
    )
    if not files:
        raise FileNotFoundError(f"no SBA_slim_part*.csv.gz in {folder}")
    dates = ["ApprovalDate", "FirstDisbursementDate", "PaidInFullDate", "ChargeOffDate"]
    frames = [pd.read_csv(f, dtype={"NaicsCode": str, "FranchiseCode": str}, parse_dates=dates, low_memory=False) for f in files]
    return pd.concat(frames, ignore_index=True)


def label(df: pd.DataFrame, asof: pd.Timestamp = ASOF, horizon: int = HORIZON) -> pd.DataFrame:
    """Keep loans with a full window and add `default` (charged off within `horizon` months of approval)."""
    d = df[df["LoanStatus"].isin(["PIF", "CHGOFF", "EXEMPT"])].copy()
    d = d[(d["GrossApproval"] > 0) & d["ApprovalDate"].notna()]
    d = d[d["ApprovalDate"] <= asof - pd.DateOffset(months=horizon)]  # window complete
    d = d[~((d["LoanStatus"] == "CHGOFF") & d["ChargeOffDate"].isna())]  # timing unknown
    end = d["ApprovalDate"] + pd.DateOffset(months=horizon)
    d["default"] = ((d["LoanStatus"] == "CHGOFF") & (d["ChargeOffDate"] <= end)).astype(int)
    d["loss"] = np.where(d["default"] == 1, d["GrossChargeOffAmount"].clip(lower=0), 0.0)
    return d.reset_index(drop=True)


def lender_history(d: pd.DataFrame, lag: int = LENDER_LAG, k: float = 20.0, col: str = "LocationID") -> pd.DataFrame:
    """A lender's record as it stood when each loan was approved: the lender's default rate and loan count over loans
    approved `lag` or more fiscal years earlier (so every one of those loans had a closed window), with the rate shrunk
    toward the all-lender rate by k loans. A lender with no history gets the all-lender rate and zero loans."""
    out = pd.DataFrame(index=d.index, columns=["lender_rate", "lender_log_loans"], dtype=float)
    if col not in d or d[col].isna().all():
        return out
    fy = d["ApprovalFY"].astype(int)
    years = np.arange(fy.min() - lag - 1, fy.max() + 1)
    n = pd.crosstab(d[col], fy).reindex(columns=years, fill_value=0).cumsum(axis=1)
    x = pd.crosstab(d[col], fy, values=d["default"], aggfunc="sum").reindex(columns=years, fill_value=0).fillna(0).cumsum(axis=1)
    tot_n, tot_x = n.sum(axis=0), x.sum(axis=0)
    v = fy - lag
    ids = d[col].astype("string").fillna("NA")
    row = pd.Index(n.index.astype("string")).get_indexer(ids)
    ci = pd.Index(years).get_indexer(v)
    ok = (row >= 0) & (ci >= 0)
    nn = np.zeros(len(d))
    xx = np.zeros(len(d))
    nn[ok] = n.to_numpy()[row[ok], ci[ok]]
    xx[ok] = x.to_numpy()[row[ok], ci[ok]]
    pool_n = np.where(ci >= 0, tot_n.to_numpy()[np.clip(ci, 0, None)], 0)
    pool_x = np.where(ci >= 0, tot_x.to_numpy()[np.clip(ci, 0, None)], 0)
    prior = np.where(pool_n > 0, pool_x / np.maximum(pool_n, 1), np.nan)
    out["lender_rate"] = (xx + k * prior) / (nn + k)
    out["lender_log_loans"] = np.log1p(nn)
    return out


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Application-time features only.

    Left out on purpose: interest rate and fixed/variable (the SBA reports them only from FY2009), and TermInMonths. The
    term is overwritten for charged-off loans (91% of loans that default have a term that is not a whole number of
    years, against 16% of the others, and the term tracks the months to charge-off), so using it would score the outcome.
    """
    x = pd.DataFrame(index=d.index)
    x["log_amount"] = np.log(d["GrossApproval"].clip(lower=1))
    x["guarantee_share"] = (d["SBAGuaranteedApproval"] / d["GrossApproval"]).clip(0, 1)
    x["jobs_per_100k"] = (d["JobsSupported"].clip(lower=0) / (d["GrossApproval"] / 1e5)).clip(upper=50)
    x[["lender_rate", "lender_log_loans"]] = lender_history(d)
    x["sector"] = d["NaicsCode"].str[:2].fillna("NA")
    x["business_type"] = d["BusinessType"].fillna("NA")
    x["age_group"] = d["BusinessAge"].map(AGE_GROUP).fillna("unknown")
    x["method"] = d["ProcessingMethod"].map(METHOD_GROUP).fillna("other")
    x["revolver"] = d["RevolverStatus"].fillna("NA")
    x["collateral"] = d["CollateralInd"].fillna("NA")
    x["franchise"] = np.where(d["FranchiseCode"].notna(), "Y", "N")
    x["state"] = d["BorrState"].fillna("NA")
    return x


def split(d: pd.DataFrame):
    fy = d["ApprovalFY"]
    train = d[(fy >= TRAIN_FY[0]) & (fy <= TRAIN_FY[1])]
    test = d[(fy >= TEST_FY[0]) & (fy <= TEST_FY[1])]
    return train, test


# ------------------------------------------------------------------ metrics
def auc(y, s) -> float:
    """Area under the ROC curve from ranks (ties get average rank); NaN if one class is missing."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ks(y, s) -> float:
    """Kolmogorov-Smirnov statistic: the largest gap between the score distributions of defaulters and non-defaulters."""
    y = np.asarray(y)
    o = np.argsort(-np.asarray(s, dtype=float), kind="stable")
    ys = y[o]
    n1 = ys.sum()
    n0 = len(ys) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float(np.max(np.abs(np.cumsum(ys) / n1 - np.cumsum(1 - ys) / n0)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y)) ** 2))


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of a score between two samples, with bins from the expected sample's quantiles."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    cuts = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)[1:-1]))
    pe = np.bincount(np.searchsorted(cuts, e, side="right"), minlength=len(cuts) + 1) / len(e)
    pa = np.bincount(np.searchsorted(cuts, a, side="right"), minlength=len(cuts) + 1) / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def calibration(y, p, bins: int = 10) -> pd.DataFrame:
    """Predicted against actual default rate by decile of predicted PD."""
    t = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    t["bin"] = pd.qcut(t["p"].rank(method="first"), bins, labels=False) + 1
    return t.groupby("bin").agg(loans=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def shift_intercept(p, target_rate: float) -> np.ndarray:
    """Move every PD by one constant on the log-odds scale so that the mean equals target_rate (a level recalibration)."""
    z = logit(p)
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return 1 / (1 + np.exp(-(z + (lo + hi) / 2)))


# ------------------------------------------------------------------ models
class Scorecard:
    """Binned logistic regression, the bank baseline. Numeric features are cut into quantile bins on the training data,
    everything becomes indicator columns, and the fitted log-odds convert to points (PDO points double the odds of default)."""

    def __init__(self, bins: int = 6, C: float = 1.0, min_share: float = 0.002):
        self.bins, self.C, self.min_share = bins, C, min_share

    def _design(self, x: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for c in self.numeric_:
            v = x[c]
            b = np.searchsorted(self.cuts_[c], v.to_numpy(dtype=float), side="right").astype(object)
            b = pd.Series(b, index=x.index).where(v.notna(), "missing").astype(str)
            cols[c] = b
        for c in self.categorical_:
            v = x[c].astype(str)
            cols[c] = v.where(v.isin(self.levels_[c]), "other")
        return pd.DataFrame(cols)

    def fit(self, x: pd.DataFrame, y) -> Scorecard:
        from sklearn.linear_model import LogisticRegression

        self.categorical_ = [c for c in CATEGORICAL if c in x.columns]
        self.numeric_ = [c for c in x.columns if c not in CATEGORICAL]
        self.cuts_ = {c: np.unique(x[c].quantile(np.linspace(0, 1, self.bins + 1)[1:-1]).to_numpy()) for c in self.numeric_}
        self.levels_ = {}
        for c in self.categorical_:
            share = x[c].astype(str).value_counts(normalize=True)
            self.levels_[c] = set(share[share >= self.min_share].index)
        design = pd.get_dummies(self._design(x), prefix_sep="=")
        self.columns_ = list(design.columns)
        self.model_ = LogisticRegression(C=self.C, max_iter=500, solver="lbfgs")
        self.model_.fit(design.to_numpy(dtype=np.float32), np.asarray(y))
        return self

    def _matrix(self, x: pd.DataFrame) -> np.ndarray:
        design = pd.get_dummies(self._design(x), prefix_sep="=").reindex(columns=self.columns_, fill_value=0)
        return design.to_numpy(dtype=np.float32)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._matrix(x))[:, 1]

    def coefficients(self) -> pd.DataFrame:
        c = pd.DataFrame({"feature": self.columns_, "log_odds": self.model_.coef_[0]})
        c[["variable", "level"]] = c["feature"].str.split("=", n=1, expand=True)
        return c.drop(columns="feature")

    def points(self, pdo: float = 20.0) -> pd.DataFrame:
        """Points table: higher points = lower risk; each `pdo` points halves the odds of default."""
        c = self.coefficients()
        c["points"] = -c["log_odds"] / math.log(2) * pdo
        return c.sort_values(["variable", "log_odds"])


def boosted(seed: int = 7):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=400,
        max_leaf_nodes=24,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        categorical_features="from_dtype",
        random_state=seed,
    )


def boosted_frame(x: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Categorical columns typed with the training categories so train and test line up."""
    out = x.copy()
    for c in CATEGORICAL:
        cats = (ref if ref is not None else x)[c].astype(str).value_counts()
        keep = list(dict.fromkeys([*cats[cats >= 50].index, "other"]))
        out[c] = pd.Categorical(out[c].astype(str).where(out[c].astype(str).isin(keep), "other"), categories=keep)
    return out


# ------------------------------------------------------------------ dollar framing
def cutoff_table(d: pd.DataFrame, pd_hat, shares=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> pd.DataFrame:
    """Approve the lowest-risk `share` of loans by count. Dollars approved, share of charged-off dollars still taken,
    default and loss rates on the approved book. Loss is the gross charge-off amount of loans charged off within the horizon."""
    t = pd.DataFrame(
        {
            "p": np.asarray(pd_hat, dtype=float),
            "amt": d["GrossApproval"].to_numpy(),
            "def": d["default"].to_numpy(),
            "loss": d["loss"].to_numpy(),
        }
    )
    t = t.sort_values("p", kind="stable").reset_index(drop=True)
    tot_loss, tot_amt, tot_def = t["loss"].sum(), t["amt"].sum(), t["def"].sum()
    rows = []
    for s in shares:
        k = int(round(len(t) * s))
        a = t.iloc[:k]
        rows.append(
            {
                "approve_share": s,
                "loans": k,
                "approved_$m": a["amt"].sum() / 1e6,
                "share_of_dollars": a["amt"].sum() / tot_amt,
                "default_rate": a["def"].mean(),
                "loss_rate_of_$": a["loss"].sum() / a["amt"].sum(),
                "defaults_avoided": 1 - a["def"].sum() / tot_def,
                "loss_$m": a["loss"].sum() / 1e6,
                "loss_avoided_share": 1 - a["loss"].sum() / tot_loss,
            }
        )
    return pd.DataFrame(rows)


def expected_loss(d: pd.DataFrame, pd_hat, lgd: float) -> float:
    """Sum of PD x LGD x exposure, in dollars."""
    return float((np.asarray(pd_hat, dtype=float) * lgd * d["GrossApproval"].to_numpy()).sum())


# ------------------------------------------------------------------ walk-forward
def fit_predict(train: pd.DataFrame, xtrain: pd.DataFrame, others: list[pd.DataFrame], gbm_iter: int = 300):
    """Fit the scorecard and the boosted model on `train`; return their PDs for each frame in `others`."""
    sc = Scorecard().fit(xtrain, train["default"])
    m = boosted()
    m.set_params(max_iter=gbm_iter)
    m.fit(boosted_frame(xtrain), train["default"])
    out = []
    for xo in others:
        out.append((sc.predict(xo), m.predict_proba(boosted_frame(xo, xtrain))[:, 1]))
    return sc, m, out


def walk_forward(
    d: pd.DataFrame, x: pd.DataFrame, first=2001, test_years=range(2011, 2022), lag: int = LENDER_LAG
) -> pd.DataFrame:
    """For each test vintage Y, fit on vintages first..Y-lag (labels complete when Y's loans were approved) and score Y."""
    rows = []
    for y in test_years:
        tr = (d["ApprovalFY"] >= first) & (d["ApprovalFY"] <= y - lag)
        te = d["ApprovalFY"] == y
        if tr.sum() < 1000 or te.sum() < 1000:
            continue
        _, _, ((ps, pg),) = fit_predict(d[tr], x[tr], [x[te]], gbm_iter=200)
        yt = d.loc[te, "default"]
        rows.append(
            {
                "vintage": y,
                "train_to": y - lag,
                "train_loans": int(tr.sum()),
                "loans": int(te.sum()),
                "default_rate": yt.mean(),
                "scorecard_auc": auc(yt, ps),
                "boosted_auc": auc(yt, pg),
                "scorecard_ks": ks(yt, ps),
                "boosted_ks": ks(yt, pg),
                "mean_pd_boosted": pg.mean(),
            }
        )
    return pd.DataFrame(rows)
