# Model validation report (M5)

Independent reproduction & adversarial review of the headline numbers. The reviewer treated the pipeline as a black box,
reproduced the headline from the pinned inputs, then tried to break it. Reproduced on commit `03d29df` (re-baselined from
`e4de697` — see note), data snapshot SHA-256 `7cce2ceba083…` (mtime 2026-03-11, unchanged), 1,444,802 rows.

> **Re-baseline (2026-06-16, commit `03d29df`).** The golden reference was re-established on the current pipeline. The
> headline moved to **risk 1.0945% / production €41,722,606 / 98 cells** (from 1.0858% / €40,685,570 / 96 cells) because of
> pipeline-code evolution since `e4de697` (audit, reject-inference and MR changes) — **not** the dependency bump:
> scikit-learn 1.9.0 was shown behaviour-neutral vs 1.8.0 by a controlled A/B (bit-identical accepted-cell set), and the
> data snapshot SHA-256 is unchanged. The reference now pins the committed standalone config
> `reports/validation/reference/no_premium_cd_config.toml` (`reject_inference_method="parceling"`, `use_mr_outcomes=true`).
> **§1's table reflects this new baseline. The §2 adversarial review has been re-run on this baseline (2026-06-16) — see
> the note under §2 — except the `no_premium_ef`/`ab`/`premium` family backtest rows in §2.2, which still require the
> full direct-portfolio batch and remain `e4de697`-vintage.**

## 1. Reproduction — PASS (deterministic)
Re-running `no_premium_cd` end-to-end (standalone config, full retrain) reproduced the committed golden reference
**byte-identically**:

| metric | reference | reproduced | Δ |
|---|---|---|---|
| risk (%) | 1.094531 | 1.094531 | **0.0 pp** |
| production (€, `oa_amt_h0`) | 41,722,606 | 41,722,606 | **0.0 %** |
| accepted cells | 98 | 98 | exact |
| accepted-set hash | 8ca147d9ed3dfb6e | 8ca147d9ed3dfb6e | match |
| data SHA-256 | 7cce2ceba083… | 7cce2ceba083… | match |

`run_reproducibility.py -s no_premium_cd` → **PASS**, `snapshot_match=True`. The pipeline is deterministic given the
pinned (data snapshot, code, config); the golden reference + SHA-256 pin makes any future drift fail loudly.

> Scope: this is the **standalone** single-segment headline (the segment trains its own model). Production shares one
> pooled `total` model across segments; the production headline is validated via the M4 backtest + the #7 multi-segment
> validation (both paths select near-identical linear models). The two are consistent in direction.

## 2. Challenge — "try to break the numbers"

> **Re-validated 2026-06-16 on the re-baselined reference (commit `03d29df`).** §2.1/2.4/2.5 were re-run from a fresh
> standalone run; §2.2's backtest was re-run for `no_premium_cd` (the re-baselined segment). Every original finding holds:
> wide risk CI, PSI stable, audit reconciles, and the out-of-time "jump" stays within sampling noise (inconclusive, thin
> window). The `no_premium_ef`/`ab`/`premium` rows in §2.2 were **not** re-run — they need the full direct-portfolio
> batch — and remain `e4de697`-vintage (marked below).

### 2.1 Bootstrap confidence interval — the headline risk is uncertain (note)
The chosen optimum's bootstrap CIs (from `risk_production_summary_table_base.csv`, "Optimum selected"):
- **Risk 1.095%**, 95% CI **[0.45%, 1.27%]** (booked-realized basis) — a **wide** band; the point estimate sits in the
  upper half. (On the blended booked+RI basis the CI is [0.76%, 1.51%].)
- **Production €41.5M**, 95% CI **[€40.4M, €43.1M]** (~±3%).

The wide risk CI says the single-snapshot risk estimate carries material sampling uncertainty — appropriate to treat the
headline risk as a band, not a point.

### 2.2 Out-of-time backtest — apparent risk drift is WITHIN SAMPLING NOISE (investigated 2026-06)
The M4 backtest *point estimates* look like upward drift for the no_premium family (on the original `e4de697` family run:
cd 0.97%→2.11%, ef 1.08%→2.62% same-basis). A dedicated investigation with confidence intervals and significance tests
shows this **does not survive scrutiny** — the headline overstated it by reporting points without CIs:

| segment | in-sample b2 (95% CI) | OOT b2 (95% CI) | OOT defaults | CI-aware flag | Poisson p |
|---|---|---|---|---|---|
| no_premium_cd *(re-run `03d29df`)* | 0.82% [0.46, 1.24] | 2.21% [0.67, 4.66] | 9 | **INCONCLUSIVE** (<10 defaults; CIs overlap) | — |
| no_premium_ef *(e4de697)* | 1.08% [0.31, 2.16] | 2.62% [0.00, 7.39] | **2** | **INCONCLUSIVE** | 0.43 (noise) |
| no_premium_ab *(e4de697)* | 1.07% [0.70, 1.51] | 1.20% [0.31, 2.43] | 9 | INCONCLUSIVE | — |
| premium *(e4de697)* | 1.19% [0.59, 1.97] | 1.41% [0.32, 3.26] | 6 | INCONCLUSIVE | — |

*`no_premium_cd` re-run on the new baseline (standalone, 2-cohort window 2025-05→2025-07, 940 booked OOT, 26% mature):
the OOT point still jumps (~2.2%) but with only **9 OOT defaults** the noise-aware rule returns **INCONCLUSIVE** (not DRIFT;
the OOT CI [0.67, 4.66] overlaps the in-sample CI [0.46, 1.24]) — the same within-noise conclusion as the original review.
The other three rows are the `e4de697` family-wide investigation, not re-executed here.*

- **Not a maturity artifact.** The `pile_h6/pile_h3` ratio is ~**1.68 identically** across in-sample (13–14 mo seasoned)
  and the OOT cohort (6–7 mo) and across all segments → the H6 pile is fully observed; `multiplier=7` correctly applied.
- **Not statistically significant.** Every OOT CI overlaps its in-sample CI; **no segment flags DRIFT** under the
  noise-aware rule (≥10 defaults AND non-overlapping CIs). On the re-baselined run `cd` has **9 OOT defaults → INCONCLUSIVE**
  (below the 10-default threshold; the earlier `e4de697` pass saw 12 defaults / borderline Poisson p=0.057); **ef was
  2 defaults — pure noise.**
- **Driven by a thin window.** The auto-derived mature window is only 2 cohorts (Jun+Jul 2025) because
  `reference = max(mis_date) ≈ 2026-01` and maturity ≥ 6 mo. Controls (ab/premium) are stable → not a pipeline artifact.

**Verdict:** inconclusive — **monitor as cohorts mature**, not a confirmed risk increase. The M4 backtest now reports
bootstrap CIs + default counts + a noise-aware DRIFT flag so thin-window noise is no longer read as drift.

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
- **Headline robustness:** the in-sample optimum is internally consistent (audit reconciles; PSI stable) but carries a
  wide risk CI. The apparent out-of-time risk drift on no_premium_cd/ef is **within sampling noise on a thin 2-cohort
  window** (§2.2) — inconclusive, not a confirmed increase; cd is borderline and worth watching.
- **Recommendation:** approved for **human-in-the-loop analyst decision-support**. Automated live cutoffs are gated on
  the `mrm_signoff.md` conditions — chiefly **re-running the M4 backtest as more cohorts mature** (the window widens past
  2 cohorts, tightening the CIs) and per-snapshot re-validation.
