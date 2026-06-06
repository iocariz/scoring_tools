# Model-risk-management sign-off (M5)

Sign-off record for the credit-risk scoring & cutoff-optimization pipeline. Complete and sign before the tool stands in
front of a regulator or drives cutoffs **automatically**. As of this milestone the tool is approved as **analyst
decision-support with a human in the loop**; automated live cutoffs require the conditions below to be met.

## 1. Model scope & description
- **Purpose:** Find optimal acceptance cutoffs on an N-dimensional score grid via MILP with monotonicity constraints,
  on a Pareto frontier of risk (`b2_ever_h6`) vs production (`oa_amt_h0`).
- **Risk metric:** `b2_ever_h6 = 7·Σ(todu_30ever_h6)/Σ(todu_amt_pile_h6)` (the `7` is the H0..H6 month count — a fixed
  constant; see the assumptions register).
- **Risk model:** selected by fresh-seed k-fold CV + 1-SE rule (audit #7); production uses one model trained on the
  pooled `total` supersegment and shared per-segment.
- **Reject inference:** parceling with the RI optimizer on (`ri_calibration_gamma` the active lever).

## 2. Key assumptions
See **`assumptions_register.md`**. Headline items: `multiplier`/`multiplier_h3` are FIXED constants (7 / 4 months);
`optimum_risk`/`risk_step` are the Core policy targets; `stress_mode`, the reject-inference knobs, and
`ri_calibration_gamma` are governed Tuning levers (materiality is **segment-dependent**, M3); `mr_maturity_months` (=6)
is the maturity anchor; `use_mr_outcomes` is validation-only (M3a).

## 3. Validation activities performed
| Activity | Artifact | Result |
|---|---|---|
| Decision-critical findings cleared (M1) | `todo-list.md` #1–#21 + sub-findings | All resolved, real-data verified |
| Data-quality hard gate (M2) | `src/data_quality.py`, lineage | Fail-closed by default; `--allow-dq-warnings` escape hatch |
| Data-snapshot pinning (M2) | `run_lineage.json` | SHA-256 / mtime / rows / git / config hash / assumptions per run |
| Config-complexity / sensitivity (M3) | `reports/sensitivity/` | Knob tiering; defaults safe; `multiplier` fixed; materiality segment-dependent |
| Out-of-time backtest (M4) | `run_backtest.py`, backtest reports | Realized vs predicted on held-out cohort |
| Reproducibility (M5) | `run_reproducibility.py`, golden reference | Headline reproduces from pinned inputs (see validation report) |
| Independent reproduction & challenge (M5) | `model_validation_report.md` | See report |
| Model-selection change (#7) | `todo-list.md` #7 + 2026-05-31 validation | **Validated** multi-segment on the pooled `total` model (pre-#7 Lasso α≈0 vs post-#7 Linear Regression — near-identical; aggregate −0.42%) |

## 4. Residual risks & limitations
1. **Out-of-time risk drift (HIGH — monitor).** The M4 backtest found realized risk on the post-training mature cohort
   ran **above** prediction for the no_premium family (no_premium_cd ~0.97%→2.11%, no_premium_ef ~1.08%→2.62% booked-only
   realized); premium/precon/ab held. Small mature windows (~2 cohorts, 21–27% mature) — a signal to **monitor**, not a
   conclusion. **Condition:** re-run the M4 backtest each period and investigate if drift persists/widens.
2. **Single data snapshot.** All numbers are from one extraction (pinned by SHA-256). **Condition:** re-validate (re-run
   the reproducibility check; it fails loudly on a new SHA) on each new snapshot.
3. **Reject-inference sensitivity.** RI levers are material where rejections are score-driven (no_premium family) and
   inert where they are not (premium). **Condition:** govern RI knobs per-segment; document any change + re-validate.
4. **Standalone vs pooled reproducibility.** The reproducibility demonstrator is a standalone single-segment run; the
   production pooled-`total` numbers are validated via M4 + #7, not by this golden check directly.
5. **Maturity / extrapolation.** Realized H6 needs ≥6-month seasoning; the most recent cohorts are necessarily excluded
   from realized backtesting (no H3 extrapolation in the M4 headline — by design).

## 5. Conditions for live / automated-cutoff use
- [ ] Reproducibility check **PASS** on the current snapshot (`run_reproducibility.py`).
- [ ] M4 backtest re-run on the latest mature cohort; the no_premium_cd/ef drift reviewed and accepted or actioned.
- [ ] Data snapshot re-validated (SHA-256 matches the signed reference, or a new reference is signed).
- [ ] Assumptions register reviewed; no FIXED constant changed; any Tuning-knob change re-validated.
- [ ] This sign-off completed by an independent validator and the model owner.

## 6. Sign-off
| Role | Name | Date | Signature |
|---|---|---|---|
| Independent validator |  |  |  |
| Model owner |  |  |  |
| Risk / governance |  |  |  |

> Status at milestone completion: **approved as human-in-the-loop analyst decision-support.** Automated live cutoffs
> remain gated on the Section 5 conditions.
