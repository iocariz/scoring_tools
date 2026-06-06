# Model validation report (M5)

Independent reproduction & adversarial review of the headline numbers. The reviewer treated the pipeline as a black box,
reproduced the headline from the pinned inputs, then tried to break it. Reproduced on commit `e4de697`, data snapshot
SHA-256 `7cce2ceba083…` (mtime 2026-03-11), 1,444,802 rows.

## 1. Reproduction — PASS (deterministic)
Re-running `no_premium_cd` end-to-end (standalone config, full retrain) reproduced the committed golden reference
**byte-identically**:

| metric | reference | reproduced | Δ |
|---|---|---|---|
| risk (%) | 1.085788 | 1.085788 | **0.0 pp** |
| production (€, `oa_amt_h0`) | 40,685,570 | 40,685,570 | **0.0 %** |
| accepted cells | 96 | 96 | exact |
| accepted-set hash | 0586adcc63302ffe | 0586adcc63302ffe | match |
| data SHA-256 | 7cce2ceba083… | 7cce2ceba083… | match |

`run_reproducibility.py -s no_premium_cd` → **PASS**, `snapshot_match=True`. The pipeline is deterministic given the
pinned (data snapshot, code, config); the golden reference + SHA-256 pin makes any future drift fail loudly.

> Scope: this is the **standalone** single-segment headline (the segment trains its own model). Production shares one
> pooled `total` model across segments; the production headline is validated via the M4 backtest + the #7 multi-segment
> validation (both paths select near-identical linear models). The two are consistent in direction.

## 2. Challenge — "try to break the numbers"

### 2.1 Bootstrap confidence interval — the headline risk is uncertain (note)
The chosen optimum's bootstrap CIs (from `risk_production_summary_table_base.csv`, "Optimum selected"):
- **Risk 1.086%**, 95% CI **[0.42%, 1.24%]** — a **wide** band; the point estimate sits in the upper half.
- **Production €40.45M**, 95% CI **[€39.4M, €42.0M]** (~±3%).

The wide risk CI says the single-snapshot risk estimate carries material sampling uncertainty — appropriate to treat the
headline risk as a band, not a point.

### 2.2 Out-of-time backtest — realized risk ran ABOVE prediction (HIGH — monitor)
The M4 backtest (`run_backtest.py`, production run, separate config) found realized risk on the post-training mature
cohort exceeding prediction for the no_premium family:
- **no_premium_cd:** predicted 1.389% → in-sample realized 0.97% → **OOT realized 2.11%** (Δ +1.14pp same-basis).
- **no_premium_ef:** 1.08% → **2.62%** (Δ +1.54pp). no_premium_ab / precon / premium held (OK).

This corroborates §2.1: newer cohorts run hotter than the in-sample estimate. Caveat: small mature windows (~2 cohorts,
21–27% mature) — a **monitoring** signal, not a conclusion. **This is the headline residual risk** (→ sign-off condition).

### 2.3 Config sensitivity (M3) — what could move the answer
- `multiplier`/`multiplier_h3` are **fixed accounting constants** (7 = H0..H6 months, 4 = H0..H3) — locked, not tunable.
- Among real levers, `stress_mode` and the reject-inference knobs are material on the no_premium family but **inert on
  premium** (95% non-score rejections); materiality is **segment-dependent** → govern per-segment. Expert/default-off
  knobs (MR suite, monotonicity, `multiplier_h3`) are inert on both segments tested.

### 2.4 Stability / drift (PSI) — stable
`drift_alerts_base.json`: **4 info, 0 warning, 0 critical** — population stability between the main and MR periods is
within tolerance (no distributional break that would invalidate the cutoffs).

### 2.5 Audit reconciliation — consistent
Loan-level `audit_base.csv` (keep / swap-in / swap-out classification) reconciles with the aggregate KPIs in the summary
table — the headline production/risk are not an aggregation artifact.

## 3. Verdict
- **Reproducible:** yes — deterministic, snapshot-pinned, with a repeatable check.
- **Assumptions:** documented and governed (`assumptions_register.md`); the one definitional constant (`multiplier`) is
  correctly locked.
- **Headline robustness:** the in-sample optimum is internally consistent (audit reconciles; PSI stable) **but** carries
  a wide risk CI and shows **upward out-of-time risk drift on no_premium_cd/ef** — the key residual risk.
- **Recommendation:** approved for **human-in-the-loop analyst decision-support**. Automated live cutoffs are gated on
  the `mrm_signoff.md` conditions — chiefly ongoing M4-backtest monitoring of the no_premium drift and per-snapshot
  re-validation.
