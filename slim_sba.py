"""
slim_sba.py -- shrink the SBA 7(a) FOIA files so they can be sent to Claude.

Run it from the Credit folder:   python slim_sba.py
It reads every 7(a) file in the folder (FOIA_7a_*, .zip or .csv, whichever is there; FY1991 to the present),
keeps only the columns the default model needs (no borrower names or street addresses), and writes
    SBA_slim_part01.csv.gz, 02, ...   loans with a known outcome (paid in full or charged off), a few MB each
    SBA_exempt_part01.csv.gz, ...     loans still active (status withheld by the SBA)
    SBA_profile.txt    row counts and missing values, so the file can be checked
The SBA files are not changed. It takes a few minutes. Do not press Ctrl+C.
"""

import pathlib

import pandas as pd

NUM = [
    "GrossApproval",
    "SBAGuaranteedApproval",
    "InitialInterestRate",
    "TermInMonths",
    "GrossChargeOffAmount",
    "JobsSupported",
    "ApprovalFY",
]
DATE = ["ApprovalDate", "FirstDisbursementDate", "PaidInFullDate", "ChargeOffDate", "AsOfDate"]
CAT = [
    "LocationID",
    "BorrState",
    "BankState",
    "ProjectState",
    "ProcessingMethod",
    "FixedorVariableInterestInd",
    "NaicsCode",
    "FranchiseCode",
    "BusinessType",
    "BusinessAge",
    "LoanStatus",
    "RevolverStatus",
    "CollateralInd",
    "SoldSecMrktInd",
    "Program",
]
KEEP = NUM + DATE + CAT


def sources():
    """One file per period: the .zip if there is one (smaller to read), else the .csv."""
    found = {}
    for p in sorted(pathlib.Path.cwd().glob("FOIA_7a_*")):
        if p.suffix.lower() in (".zip", ".csv") and (p.stem not in found or p.suffix.lower() == ".zip"):
            found[p.stem] = p
    if not found:
        raise SystemExit(f"No FOIA_7a_* files found in {pathlib.Path.cwd()}")
    return list(found.values())


def tidy(chunk):
    for c in NUM:
        if c in chunk:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("float32")
    for c in DATE:
        if c in chunk:
            chunk[c] = pd.to_datetime(chunk[c], errors="coerce")
    for c in CAT:
        if c in chunk:
            chunk[c] = chunk[c].astype("string").str.strip().astype("category")
    if "LoanStatus" in chunk:  # the file spells paid in full "P I F"
        chunk["LoanStatus"] = chunk["LoanStatus"].astype("string").str.replace(" ", "", regex=False).astype("category")
    return chunk


frames, log = [], []
for src in sources():
    print("reading", src.name, flush=True)
    n = 0
    for chunk in pd.read_csv(
        src, usecols=lambda c: c in KEEP, dtype=str, chunksize=200_000, encoding="latin-1", low_memory=False
    ):
        frames.append(tidy(chunk))
        n += len(chunk)
        print("  rows", n, flush=True)
    log.append(f"{src.name}: {n} rows")

print("joining...", flush=True)
cats = {c: pd.api.types.union_categoricals([f[c] for f in frames if c in f]).categories for c in CAT}
for f in frames:
    for c in CAT:
        if c in f:
            f[c] = pd.Categorical(f[c].astype(object), categories=cats[c])
df = pd.concat(frames, ignore_index=True)
missing = [c for c in KEEP if c not in df]
print("rows", len(df), "columns", len(df.columns), "missing columns:", missing, flush=True)

lines = log + [f"missing columns: {missing}", f"rows: {len(df)}", ""]
lines.append("LoanStatus counts:\n" + df["LoanStatus"].value_counts(dropna=False).to_string())
lines.append("\nApprovalFY x LoanStatus:\n" + pd.crosstab(df["ApprovalFY"], df["LoanStatus"]).to_string())
lines.append("\nmissing share by column:\n" + df.isna().mean().round(4).to_string())
for c in (
    "ProcessingMethod",
    "BusinessType",
    "BusinessAge",
    "FixedorVariableInterestInd",
    "CollateralInd",
    "SoldSecMrktInd",
    "RevolverStatus",
    "Program",
):
    if c in df:
        lines.append(f"\n{c}:\n" + df[c].value_counts(dropna=False).head(15).to_string())
lines.append("\nAsOfDate: " + str(df["AsOfDate"].dropna().unique()[:3]))
lines.append("numeric summary:\n" + df[[c for c in NUM if c in df]].describe().T.to_string())
pathlib.Path("SBA_profile.txt").write_text("\n".join(lines), encoding="utf-8")

# Parquet needs pyarrow, which Windows Application Control blocks on some machines, so write plain gzipped CSV parts.
# Only loans with a known outcome (paid in full or charged off) are needed for the default model; the profile above
# was computed on all loans. Parts are small so that each one transfers quickly.
for old in pathlib.Path.cwd().glob("SBA_slim_part*.csv.gz"):
    old.unlink()
drop = ["Program", "AsOfDate"]
out = df[df["LoanStatus"].isin(["PIF", "CHGOFF"])].drop(columns=drop).sort_values("ApprovalDate", kind="stable")
print("loans with a known outcome:", len(out), flush=True)
size = 150_000
total = 0.0
for k, start in enumerate(range(0, len(out), size), start=1):
    name = f"SBA_slim_part{k:02d}.csv.gz"
    out.iloc[start : start + size].to_csv(name, index=False, compression="gzip", date_format="%Y-%m-%d", float_format="%.6g")
    mb = pathlib.Path(name).stat().st_size / 1e6
    total += mb
    print(name, round(mb, 1), "MB", flush=True)
print("total", round(total, 1), "MB")

# Loans that are disbursed and still active are EXEMPT in this file (status withheld). They have not been charged off,
# so the default model needs them as loans that have not defaulted yet.
for old in pathlib.Path.cwd().glob("SBA_exempt_part*.csv.gz"):
    old.unlink()
act = df[df["LoanStatus"] == "EXEMPT"].drop(columns=drop).sort_values("ApprovalDate", kind="stable")
print("active (EXEMPT) loans:", len(act), flush=True)
for k, start in enumerate(range(0, len(act), size * 2), start=1):
    name = f"SBA_exempt_part{k:02d}.csv.gz"
    act.iloc[start : start + size * 2].to_csv(name, index=False, compression="gzip", date_format="%Y-%m-%d", float_format="%.6g")
    print(name, round(pathlib.Path(name).stat().st_size / 1e6, 1), "MB", flush=True)
print("Done. Tell Claude.")
