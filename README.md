# SBA 7(a) probability-of-default model

A default model for small-business loans, built from the SBA's public loan file (1.7 million 7(a) loans, FY1991 to
FY2026), tested the way a bank validates a model: train on the past, score the future, and report what holds up. Version 2 refits
the model every year on loans whose outcomes were already known (pooled AUC 0.69 on FY2016 to FY2021 approvals, against 0.64
for loan size alone), and the document records a leaked loan-term field that on its own reaches the AUC (0.98) a published preprint reports.

**Read `MODEL_DOCUMENT.md` first.** It opens with the decision, then the evidence.

| File | What it is |
|---|---|
| `MODEL_DOCUMENT.md` | The model document: purpose, data, label, method, validation, dollars, limits, monitoring |
| `sba_pd.py` | Label, features, scorecard, boosted model, metrics, dollar tables |
| `run_analysis.py` | Version 1 (one model frozen at 2009): reproduces its numbers into `results/` |
| `run_refit.py` | Version 2 (yearly refit), the interest-rate audit and the term-leak replication, into `results/refit_*.csv` |
| `check_doc.py` | Checks that every figure quoted in the document matches `results/` |
| `slim_sba.py` | Run once on the SBA download folder; keeps the columns the model needs and drops borrower names and addresses |
| `tests/` | Hand-computed checks of the label, the lender record, the metrics and the dollar table |
| `results/` | The tables behind the document |

Data: SBA "7(a) & 504 FOIA" files (data.sba.gov/dataset/7a-504-foia), as of 2026-09-30. Not included here; the files
are large. Run `python slim_sba.py` in the folder holding the `FOIA_7a_*` downloads, copy the `SBA_*.csv.gz` files into
`data/`, then `python run_analysis.py data` (about 5 minutes) and `python run_refit.py data` (about 1 minute). `python check_doc.py` and `pytest` check the document and the code.
