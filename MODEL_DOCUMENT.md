# SBA 7(a) probability-of-default model: model document

Kelvin Afful · built November 2026 · code and tables in this repository · data: SBA 7(a) FOIA file as of 30 September 2026

## 1. Decision

**Use the model to rank new applications and send the riskiest fifth to enhanced review, and refit it every year. Do not price
loans or set provisions on its default probabilities, and do not decline on the score alone.**

Two versions are reported. **Version 2 (yearly refit)** is the one to use: each year's model is trained only on loans approved at least six
years earlier, whose five-year outcomes were already known. **Version 1 (frozen at 2009)** is one model trained on FY2001 to FY2009 and left alone; it is kept as a robustness check and for what it shows about regime change.

- **It ranks, but modestly.** On 294,362 loans approved FY2016 to FY2021 (11,018 defaults), version 2 has an AUC of 0.692 (95% interval
  0.688 to 0.696) and a KS of 0.283. The riskiest tenth of loans defaulted at 8.5% within five years and the safest tenth at 0.45%.
- **Most of the signal is still loan size.** Ranking by size alone scores 0.643 on the same loans. The version 1 fields refit each year
  score 0.662; adding the interest rate lifts that by about 0.025 and place fields (lender or project outside the borrower's state) by about 0.005, 0.027 to 0.033 in all.
- **The level is close once the model is refit.** Mean predicted default rate is 3.2% against 3.7% actual, with no shift applied. The middle
  deciles are too low (decile 6: 2.8% predicted, 3.9% actual) and the top is slightly high. The Brier score is 0.0355 against 0.0360 for quoting the base rate to everyone, so the probabilities carry little
  beyond the ranking.
- **In dollars** (FY2016 to FY2021 approvals, $139.7 billion): sending the riskiest 20% of applications by count to review covers 40% of the defaults and 23% of the
  charged-off dollars while touching 5% of approved dollars. Only 26% of gross charge-off dollars ($372 million of $1,451 million) are the unguaranteed part, so a
  lender's own exposure is about a quarter of the gross figure before recoveries, which the public file does not report.

A field that leaked the outcome was found and removed (section 4). With it the boosted model scored 0.93, which is the
number a careless build would have reported.

### Version 1: one model frozen at 2009

- **It ranks, but modestly.** On loans approved after the model was built, the boosted model has an AUC of 0.65
  (95% interval 0.647 to 0.656) and the logistic scorecard 0.61. The riskiest tenth of loans defaulted at 7.2% within five years and the safest
  tenth at 1.0%.
- **Most of the signal is loan size.** Smaller loans default more. Ranking by size alone scores 0.639 on the same loans, so the
  boosted model adds 0.013 and the scorecard scores below size alone (0.614). Everything else in the public file (industry, state,
  lending program, business age, jobs) adds little once size is known.
- **The probabilities are wrong in level (version 1 only; a yearly refit fixes most of it, section 13).** The model learned from 2001 to 2009 approvals, when 15% of loans defaulted within
  five years (28% for 2007 approvals). Later loans defaulted at 3.7%. Raw model PDs average 13.4% against 3.7% actual. Shifting
  every PD by one constant (fitted on FY2010 to FY2014 outcomes) fixes the average (3.3% against 3.8% for FY2015 to FY2021)
  but not the individual estimates: the Brier score after the shift (0.0363) equals that of quoting the base rate to everyone (0.0362).
- **In dollars** (FY2015 to FY2021 approvals, $160.1 billion): sending the riskiest 20% of applications by count to review covers
  34% of the defaults and 19% of the charged-off dollars while touching 4% of approved dollars. The scorecard covers 30% and 13%.
  Charged-off dollars are gross of the SBA guarantee and of recoveries; they are not a lender's loss.

| | Scorecard | Boosted | Size alone |
|---|---|---|---|
| AUC, approvals FY2010 to FY2021 (about 20,900 defaults) | 0.622 | 0.653 | 0.640 |
| AUC, approvals FY2015 to FY2021 (13,150 defaults) | 0.614 | 0.652 | 0.639 |
| KS, FY2015 to FY2021 | 0.172 | 0.231 | |
| AUC on the training years | 0.716 | 0.774 | |

## 2. Purpose and scope

The model answers one question at approval: **will this 7(a) loan be charged off within 60 months?** It uses what is
recorded at approval, with no credit score, financial statements, owner equity or collateral value, because the public file has none of those. It does not estimate loss given default, and it does not cover the SBA 504 program.

## 3. Data and label

- **Source.** SBA 7(a) FOIA loan files (FY1991 to FY2026), 1,976,761 loans. Cancelled and undisbursed loans (267,150) are not
  outcomes and are dropped, leaving 1,709,611 paid in full (1,189,078), charged off (222,294) and still active (298,239).
  The SBA labels still-active loans `EXEMPT`: disbursed, not cancelled, paid or charged off.
- **Label.** Default = charged off within 60 months of approval. Paid-in-full and still-active loans are not defaults.
  Charge-offs take a median of 51 months, so "ever charged off" would flatter old loans and punish recent ones; a fixed window treats every vintage alike.
- **Population.** Loans approved at least 60 months before the file date (approved by 30 September 2021), so every loan has a
  complete window: 1,461,966 loans. A charge-off after month 60 is not a default here. 56 charge-offs with no date, and 3 loans with a non-positive amount, are dropped.
- **Regimes.** Default rates by approval year run 6% in 2001, 13% in 2005, 22% in 2006, 28% in 2007, 21% in 2008, 8% in 2009, then 3% to 5% from
  2010 on. The training years sit on the financial crisis; the test years do not.

## 4. Leakage controls

Fields describing the outcome (status, charge-off amount and date, paid-in-full date, secondary-market sale) are never
features. One field looked innocent and was not: **`TermInMonths`**. In loans approved FY2010 to FY2016, 9% of loans that defaulted have a
term in whole years (84, 120 and so on) against 84% of the others, and for defaulted loans the term tracks the months to
charge-off. The SBA appears to overwrite the term when a loan is charged off. A boosted model using it scored AUC 0.93
on later loans; without it, 0.65. The term is excluded. Version 1 also leaves out the interest rate and fixed/variable flag, which the SBA
reports only from FY2009 on (a feature that is missing for the whole training window cannot be learned); version 2 trains on FY2010 onward and includes them after the audit in section 13.

## 5. Method

Sections 5 to 9 describe version 1 (one model frozen at 2009); version 2 is section 13.

- **Features (11), all known at approval.** Loan amount (log), SBA-guaranteed share, jobs supported per $100,000, industry (two-digit NAICS), state,
  business type, business age (grouped), processing program (grouped), revolving line, collateral flag, franchise flag.
- **Split.** Train on FY2001 to FY2009 approvals (570,452 loans; industry codes exist from FY2001), test on FY2010 to FY2021
  (560,588). Because five-year outcomes of FY2009 loans were only known by late 2014, **FY2015 to FY2021 is the real-time test** (349,782 loans): every training label was
  known before those loans were approved.
- **Scorecard (the bank baseline).** Numbers cut into six quantile bins, everything one-hot, ridge logistic regression; the fitted
  log-odds convert to points (20 points halve the odds of default; `results/scorecard_points.csv`).
- **Boosted model.** Gradient boosted trees (scikit-learn), fixed settings, early stopping on a random tenth of the training years.
- **Nothing was tuned on the test years.** Settings were fixed before the run. The choices made after seeing results are listed in section 10.

## 6. Does the ranking hold?

- **By approval year** (fixed model, `results/by_vintage.csv`): boosted AUC between 0.630 and 0.696 in every year from 2010 to 2021;
  the scorecard between 0.592 and 0.692. Neither decays steadily with time.
- **Walk-forward** (`results/walk_forward.csv`): for each year, refit on loans approved at least six years earlier and score that year.
  Boosted AUC runs from 0.61 (2011, trained only on FY2001 to FY2005) to 0.73 (2021), and is 0.63 to 0.69 for 2013 to 2020. The scorecard is weaker
  and less stable (0.54 to 0.72).
- **Sampling error.** On FY2015 to FY2021 the 95% intervals are 0.647 to 0.656 (boosted) and 0.609 to 0.618 (scorecard); the gap between them is 0.034 to 0.041, so it is not noise.
- **Placebo.** With default labels shuffled within each approval year the boosted AUC is 0.506 (sd 0.0015), so nothing in the pipeline produces a result on its own.
- **Ablations** (scorecard, same split; `results/ablation_scorecard.csv`):

| Specification | AUC FY2010 to FY2021 | AUC FY2015 to FY2021 |
|---|---|---|
| Loan size alone, monotone | 0.640 | 0.639 |
| All features | 0.622 | 0.614 |
| Without state | 0.609 | 0.596 |
| Without state or industry | 0.599 | 0.587 |
| Loan size alone, six bins | 0.625 | 0.625 |
| All features, twelve bins | 0.632 | 0.627 |

  Binning throws away ordering within a bin, which is why six bins of size score below size ranked directly. State adds 0.013 to 0.019 and industry about 0.010, but not enough to pass size alone.

## 7. Calibration and stability

- **Level.** The boosted model's deciles of predicted PD run from 1.7% to 37.6%; actual default rates in the test years run from 1.0% to 7.2% (`results/calibration_boosted.csv`). Actual rates
  rise at every step, so the order is right and the scale is not.
- **Recalibration** by one constant on the log-odds scale, fitted on FY2010 to FY2014 outcomes (shift of -1.58) and applied to FY2015 to FY2021: mean PD 3.3% against 3.8% actual, Brier 0.0363
  (raw 0.0515; base rate alone 0.0362). Aggregate expected loss (PD x 77% loss given default x approved dollars) is $1.75 billion against $1.69 billion actual, 4% high. The shift has to be refitted whenever the regime changes.
- **Stability.** The population stability index of the boosted score against the training years is 0.04 to 0.16 for 2010 to 2020 approvals and 0.78 for 2021, when the mix
  of loans changed (larger loans, mean PD 7.6%); a value above 0.25 means the population has moved.

## 8. Drivers

Boosted model, permutation importance (fall in AUC when one feature is shuffled; `results/drivers_boosted.csv`): loan amount 0.098, state 0.027,
processing program 0.025, industry 0.018, business age 0.010, jobs per dollar 0.008, guaranteed share 0.007, all others under 0.003. In the scorecard, loan size, state, program and industry each span more than 30 points
of the score; collateral, business age and jobs span 9 to 12. Smaller loans score as riskier, loans through the certified and preferred lender programs score safer, and the "other" programs score riskiest (the points are in `results/scorecard_points.csv`). Some directions run against intuition: established businesses and loans with a collateral flag score slightly riskier than startups and loans without one. That is a reason to read the scorecard as a fit to the training years and not as underwriting rules; these are associations in a program with its own selection, not causes.

## 9. Dollars

Approve the lowest-risk share of FY2015 to FY2021 applications (`results/cutoffs_boosted.csv`). Loss is the gross charge-off amount within 60 months.

| Approve | Share of approved dollars kept | Default rate | Loss rate of dollars | Defaults avoided | Charged-off dollars avoided |
|---|---|---|---|---|---|
| 100% | 100% | 3.8% | 1.05% | 0% | 0% |
| 90% | 98% | 3.4% | 0.98% | 18% | 9% |
| 80% | 96% | 3.1% | 0.89% | 34% | 19% |
| 70% | 93% | 2.8% | 0.82% | 48% | 28% |
| 60% | 90% | 2.5% | 0.74% | 60% | 37% |
| 50% | 85% | 2.1% | 0.65% | 72% | 47% |

The riskiest loans are small, so cutting by count removes few dollars. Referral is cheap at the 20% level, and the benefit is a fifth of
charged-off dollars, not most of them. The SBA guarantee (median 75% of the loan) and recoveries sit between these dollars and a lender's loss.

## 10. Tested and rejected, and choices made after seeing results

- **Lender record.** A lender's own default rate and loan count, taken only from loans approved six or more years earlier, lowered the test AUC (boosted 0.637 against 0.653, scorecard 0.619 against 0.622). A lender's past does not carry over across regimes. It is in the code (`LENDER`) and not in the model.
- **Term of the loan:** excluded in both versions (section 4). **Interest rate:** excluded in version 1, included in version 2 after an audit (section 13). **Disbursement lag** (a field known only after approval) and **four-digit industry code** were tried and left out; the industry code lowered AUC.
- The choice to drop the lender record, and the ablations in section 6, were made after the first full run; they are diagnostics, not tuned results.

## 11. Limits

- The file covers SBA-guaranteed loans, which lenders chose to put in the program; it says nothing about conventional small-business lending.
- No applicant credit score, cash flow or collateral value. A bank would add them, and they would carry much of the signal that is missing here.
- The label stops at 60 months. Later charge-offs count as survivors, and loans approved in 2020 and 2021 are mostly still active.
- Test years include the 2020 to 2021 pandemic program changes. The 2021 vintage looks different (section 7) and is reported, not removed.
- Charge-off is not economic loss. Dollars here are gross.
- The interest-rate audit (section 13) finds no sign that the field is rewritten at charge-off, but it cannot prove it; if the SBA changed it after charge-off, the 0.025 gain from the rate would be partly leakage.
- The model was built to be checked, not deployed: nothing here has been run on a live pipeline.

## 12. Monitoring if it were used

Quarterly: AUC and KS on each matured vintage; population stability of the score (investigate above 0.10, redevelop above 0.25); actual against predicted
default rate by decile, with the level shift refitted every year. Redevelop if AUC falls below 0.60 on a vintage of 30,000 or more
loans, or if the ranking of the riskiest decile against the safest narrows below 3 to 1 (it is 7 to 1 for version 1 and 19 to 1 for version 2).

## 13. Version 2: yearly refit, added fields and audit

- **Design.** For each approval year Y from 2016 to 2021, fit the boosted model on loans approved FY2010 to Y minus 6 and score Y. FY2010 is the first training year because the interest rate is reported from FY2009. The 2016 model has only FY2010 loans to learn from; the 2021 model has FY2010 to FY2015.
- **Variants** (`results/refit_pooled.csv`, `results/refit_by_vintage.csv`):

| Fields | Pooled AUC, FY2016 to FY2021 | KS |
|---|---|---|
| Version 1 fields, refit each year | 0.662 | 0.234 |
| Plus lender or project outside the borrower's state | 0.667 | 0.239 |
| Plus initial interest rate and variable-rate flag | 0.692 | 0.283 |

  The 95% interval for the full model is 0.688 to 0.696, and for its gain over the version 1 fields 0.027 to 0.033. The place fields help in some years and hurt in 2016, so their 0.005 is small. The interest-rate gain shows in all six years.
- **Calibration** (`results/refit_calibration.csv`): decile of predicted PD runs from 0.5% to 9.5%, actual from 0.4% to 8.5%. Mean PD 3.2% against 3.7%, with no shift. The model underpredicts the middle deciles.
- **Dollars** (`results/refit_cutoffs.csv`), approving the lowest-risk share by count:

| Approve | Share of approved dollars kept | Default rate | Defaults avoided | Gross charge-offs avoided | Unguaranteed charge-offs avoided |
|---|---|---|---|---|---|
| 100% | 100% | 3.7% | 0% | 0% | 0% |
| 90% | 98% | 3.2% | 23% | 12% | 12% |
| 80% | 95% | 2.8% | 40% | 23% | 23% |
| 70% | 92% | 2.4% | 55% | 34% | 35% |
| 60% | 87% | 2.1% | 67% | 44% | 46% |
| 50% | 82% | 1.7% | 77% | 55% | 57% |

  Unguaranteed charge-offs are gross charge-offs times one minus the loan's SBA-guaranteed share. They are an upper bound on a lender's loss, because recoveries are not in the file.
- **Audit of the interest rate.** The term was caught because defaulted loans carried a rewritten, non-round value. The rate shows none of that: it is present for every loan (0% missing for defaulted and other loans), its precision is the same (0.2% of defaulted and 0.4% of other loans carry a third decimal), and defaulted variable-rate loans carry higher initial rates than others in the same year (8.2% against 7.4% for FY2018), the direction risk-based pricing predicts. Among defaulted loans, those charged off within 12 months of approval carry a higher rate (8.3%) than those charged off after 48 months (7.1%), which fits riskier loans failing sooner. None of this proves the field is untouched.
- **Audit of the 18-month early-default design** (`run_early_audit.py`, `results/early_audit.csv`, `results/facts_early.json`). The same population as the replication in section 14: 119,337 test loans, 378 charged off within 18 months, trained on FY2010 to FY2019 (1,334 early defaults). Without the term, the boosted model scores an AUC of 0.817 (95% interval 0.798 to 0.834), against 0.980 with the term alone, so the term is the only field that carries a leaked signal. Loan size alone scores 0.775 and removing it costs 0.050; removing any other single field costs at most 0.019. Early defaults are small loans: a median of $50,000 in the test window against $165,600 for the rest ($25,000 against $125,000 in training). That is a real pattern in this file, not proof of one, because nothing here shows how the SBA records the amount at charge-off. Two cautions: 378 defaults is a small base, and a boosted model given the term and every other field scored lower (0.770) than the term alone, so the early design is noisy even before leaks. The five-year design in the rest of this document does not use it.

## 14. Related work

- Glennon and Nigro (2005), *Journal of Money, Credit and Banking* 37(5), 923-47, is the standard published study of SBA default risk, using a survival model on older loans.
- Ahor (2025), an SSRN preprint (no. 7197485) by an independent researcher, ranks early default (charge-off within 18 months) on the same FOIA file and reports a test AUC of 0.975 for XGBoost, with loan term the dominant feature, and a term-only model that still ranks strongly. The paper does not test whether the term is rewritten at charge-off. In this file it is. Among loans approved from October 2021 to March 2024 (119,337 loans, 378 charged off within 18 months), 14% of the early defaults have a term in whole years against 86% of the other loans, and a boosted model trained on FY2010 to FY2019 using the term alone scores an AUC of 0.980 on them (`results/facts_refit.json`). That reproduces the preprint's headline number from the leaked field alone, so results that rest on the term should be read with that in mind. This is a replication of the design on a later file date, not a re-run of the paper's code.
- Shumway (2001), *Journal of Business* 74(1), 101-24, and Campbell, Hilscher and Szilagyi (2008), *Journal of Finance* 63(6), 2899-2939, motivate out-of-time testing and market-based inputs in default models; this document applies the out-of-time logic to loans.

## 15. Reproduce

`python slim_sba.py` in the folder with the SBA downloads, copy the `SBA_*.csv.gz` files into `data/`, `python run_analysis.py data`
(about 5 minutes, version 1), `python run_refit.py data` (about 1 minute, version 2), `python run_early_audit.py data` (about 2 minutes), `python check_doc.py`, `pytest` (tests in `tests/`). Every number above is in `results/`.
