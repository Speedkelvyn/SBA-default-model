# SBA 7(a) probability-of-default model

A default model for small-business loans, built from the SBA's public loan file (1.7 million 7(a) loans, FY1991 to
FY2026), tested the way a bank validates a model: train on the past, score the future, and report what holds up.

**Read `MODEL_DOCUMENT.md` first.** It opens with the decision, then the evidence.

| File | What it is |
|---|---|
| `MODEL_DOCUMENT.md` | The model document: purpose, data, label, method, validation, dollars, limits, monitoring |
| `sba_pd.py` | Label, features, scorecard, boosted model, metrics, dollar tables |
| `run_analysis.py` | Reproduces every number in the document into `results/` |
| `slim_sba.py` | Run once on the SBA download folder; keeps the columns the model needs and drops borrower names and addresses |
| `tests/` | Hand-computed checks of the label, the lender record, the metrics and the dollar table |
| `results/` | The tables behind the document |

Data: SBA "7(a) & 504 FOIA" files (data.sba.gov/dataset/7a-504-foia), as of 2026-09-30. Not included here; the files
are large. Run `python slim_sba.py` in the folder holding the `FOIA_7a_*` downloads, copy the `SBA_*.csv.gz` files into
`data/`, then `python run_analysis.py data` (about 5 minutes). `pytest` runs the tests.
