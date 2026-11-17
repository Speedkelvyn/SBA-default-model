# Copyright (c) 2026 Kelvin Afful. MIT License, see LICENSE.
"""Check that the figures quoted in MODEL_DOCUMENT.md match results/ (run after run_analysis.py)."""

import json

import pandas as pd

with open("MODEL_DOCUMENT.md", encoding="utf-8") as fh:
    doc = fh.read()
with open("results/facts.json") as fh:
    f = json.load(fh)
perf = pd.read_csv("results/performance.csv").set_index("model")
abl = pd.read_csv("results/ablation_scorecard.csv").set_index("spec")
wl = pd.read_csv("results/with_lender_record.csv").set_index("model")
cb = pd.read_csv("results/cutoffs_boosted.csv").set_index("approve_share")
cs = pd.read_csv("results/cutoffs_scorecard.csv").set_index("approve_share")
cal = pd.read_csv("results/calibration_boosted.csv")
rec = pd.read_csv("results/recalibration.csv").iloc[0]
vin = pd.read_csv("results/vintages.csv").set_index("ApprovalFY")
bv = pd.read_csv("results/by_vintage.csv")
wf = pd.read_csv("results/walk_forward.csv")
drv = pd.read_csv("results/drivers_boosted.csv").set_index("feature")

checks = []


def has(label, text):
    checks.append((label, text in doc, text))


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


has("boosted strict AUC", f"{perf.loc['boosted', 'strict_test_auc']:.3f}")
has("boosted test AUC", f"{perf.loc['boosted', 'test_auc']:.3f}")
has("scorecard test AUC", f"{perf.loc['scorecard', 'test_auc']:.3f}")
has("scorecard strict AUC", f"{perf.loc['scorecard', 'strict_test_auc']:.3f}")
has("train AUCs", f"{perf.loc['scorecard', 'train_auc']:.3f}")
has("train AUC boosted", f"{perf.loc['boosted', 'train_auc']:.3f}")
has("KS", f"{perf.loc['boosted', 'strict_test_ks']:.3f}")
has("KS scorecard", f"{perf.loc['scorecard', 'strict_test_ks']:.3f}")
has("size only", f"{f['auc_size_only_strict']:.3f}")
has("CI boosted", f"{f['boot_boosted_auc_ci'][0]:.3f} to {f['boot_boosted_auc_ci'][1]:.3f}")
has("CI scorecard", f"{f['boot_scorecard_auc_ci'][0]:.3f} to {f['boot_scorecard_auc_ci'][1]:.3f}")
has("CI gap", f"{f['boot_gap_ci'][0]:.3f} to {f['boot_gap_ci'][1]:.3f}")
has("strict defaults", f"{f['strict_defaults']:,}")
has("placebo", f"{f['placebo_auc_mean']:.3f} (sd {f['placebo_auc_sd']:.4f})")
has("population", f"{f['population']:,}")
has("train loans", f"{f['train_loans']:,}")
has("test loans", f"{f['test_loans']:,}")
has("strict loans", f"{f['strict_loans']:,}")
has("loans in file", f"{f['statuses']['PIF'] + f['statuses']['CHGOFF'] + f['statuses']['EXEMPT']:,}")
for k, v in f["statuses"].items():
    has(f"status {k}", f"{v:,}")
has("train rate", pct(f["train_rate"]))
has("test rate", pct(f["test_rate"], 1))
has("mean pd raw", pct(perf.loc["boosted", "mean_pd_test"], 1))
has("recal mean", pct(rec["mean_pd_recalibrated"], 1))
has("recal actual", pct(rec["actual"], 1))
has("brier recal", f"{rec['brier_recalibrated']:.4f}")
has("brier raw", f"{rec['brier_raw']:.4f}")
has("brier const", f"{f['brier_constant_strict']:.4f}")
has("shift", f"{f['recal_shift_logodds']:.2f}")
has("lgd", pct(f["lgd_train"]))
has("approved $bn", f"${f['approved_$m'] / 1000:.1f} billion")
has("expected loss", f"${f['expected_loss_boosted_$m'] / 1000:.2f} billion")
has("actual loss", f"${f['actual_loss_$m'] / 1000:.2f} billion")
has("term defaulted", pct(f["term_whole_years_defaulted"]))
has("term other", pct(f["term_whole_years_other"]))
has("decile low/high pred", f"{pct(cal.predicted.iloc[0], 1)} to {pct(cal.predicted.iloc[-1], 1)}")
has("decile actual high", pct(cal.actual.iloc[-1], 1))
has("decile actual low", pct(cal.actual.iloc[0], 1))
for y in (2001, 2005, 2006, 2007, 2008, 2009):
    has(f"vintage {y}", pct(vin.loc[y, "default_rate"]))
for s in (0.9, 0.8, 0.7, 0.6, 0.5):
    r = cb.loc[s]
    has(f"cut {s} dollars", pct(r["share_of_dollars"]))
    has(f"cut {s} loss rate", f"{100 * r['loss_rate_of_$']:.2f}%")
    has(f"cut {s} avoided", pct(r["defaults_avoided"]))
    has(f"cut {s} loss avoided", pct(r["loss_avoided_share"]))
has("scorecard cut 0.8", pct(cs.loc[0.8, "defaults_avoided"]))
has("scorecard cut 0.8 loss", pct(cs.loc[0.8, "loss_avoided_share"]))
has("ablation no state", f"{abl.loc['scorecard, no state', 'test_auc']:.3f}")
has("ablation no state/sector", f"{abl.loc['scorecard, no state or sector', 'strict_test_auc']:.3f}")
has("ablation 12 bins", f"{abl.loc['scorecard, 12 bins per number', 'test_auc']:.3f}")
has("lender boosted", f"{wl.loc['boosted + lender record', 'test_auc']:.3f}")
has("lender scorecard", f"{wl.loc['scorecard + lender record', 'test_auc']:.3f}")
has("vintage auc range boosted", f"{bv.boosted_auc.min():.3f} and {bv.boosted_auc.max():.3f}")
has("vintage auc range scorecard", f"{bv.scorecard_auc.min():.3f} and {bv.scorecard_auc.max():.3f}")
has("walk-forward low", f"{wf.boosted_auc.min():.2f}")
has("walk-forward high", f"{wf.boosted_auc.max():.2f}")
for k in ("log_amount", "state", "method", "sector", "age_group", "jobs_per_100k", "guarantee_share"):
    has(f"driver {k}", f"{drv.loc[k, 'auc_drop']:.3f}")
psi = bv.psi_boosted_vs_train[bv.vintage <= 2020]
has("psi range", f"{psi.min():.2f} to {psi.max():.2f}")
has("psi 2021", f"{bv.psi_boosted_vs_train.iloc[-1]:.2f}")
has("2021 mean pd", pct(bv.mean_pd_boosted.iloc[-1], 1))

bad = [c for c in checks if not c[1]]
for lab, ok, text in checks:
    if not ok:
        print("MISSING:", lab, "->", text)
print(f"{len(checks) - len(bad)} of {len(checks)} figures found in the document")
raise SystemExit(1 if bad else 0)
