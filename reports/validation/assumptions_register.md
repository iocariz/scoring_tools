# Assumptions register (M5)

The load-bearing modelling assumptions behind the headline numbers, with current values, source, rationale, governance
tier, and who may change them. Tiers are from the M3 config-complexity audit (`reports/sensitivity/`):
**FIXED** (definitional constant — never tune), **Core** (must be set; defines the problem), **Tuning** (governed lever
with material headline impact), **Expert** (fragile / validation-only / default-off).

Snapshot of `config.toml` at time of writing (the per-run truth is captured in `run_lineage.json`, M2).

| Parameter | Value | Default | Source | Tier | Rationale / notes | Who may change |
|---|---|---|---|---|---|---|
| `multiplier` | 7 | 7.0 | `config.py` (multiplier field) | **FIXED** | `todu_amt_pile_h6` is the **sum** of outstanding over months **H0..H6 = 7 months**; the `7` in `b2_ever_h6 = 7·Σ(todu_30ever_h6)/Σ(todu_amt_pile_h6)` converts that sum to the average monthly outstanding. **Not a tuning knob** — its large sensitivity-sweep impact is a guardrail signal (changing it mis-scales the denominator). | Nobody (locked to the metric definition) |
| `multiplier_h3` | 4 | 4.0 | `config.py` | **FIXED** | Same logic, H0..H3 = 4 months. | Nobody |
| `optimum_risk` | per-segment (`segments.toml`; e.g. no_premium_cd = 1.40) | 1.1 | `config.py` / `segments.toml` | **Core** | The risk target (%) the optimizer selects against. Defines the policy. | Model owner / portfolio policy |
| `risk_step` | 0.1 | 0.1 | `config.py` | **Core** | Pareto-frontier resolution (%). | Model owner |
| `variables` / `inference_variables` / `bins` / `directions` | per-config | — | `config.toml` | **Core** | The score grid, model features, bin edges, and monotonicity. Bin edges are **frozen per run** (learned once on the demand population) and must be reused to reproduce. | Model owner |
| `stress_mode` | `disabled` | `global` | `config.py` | **Tuning** | Stress factor mode. Material (±29% on no_premium in M3) **but segment-dependent**. `disabled` avoids stress×parceling double-counting (guardrail at `pipeline/preprocessing.py`). | Governed change + re-validation |
| `reject_inference_method` | `parceling` | `none` | `config.py` | **Tuning** | Selection-bias correction for swap-in candidates. | Governed |
| `reject_parceling_method` | `linear` | `linear` | `config.py` | **Tuning** | RI acceptance-rate shape. Material on no_premium (±10%), **inert on premium** (95% non-score rejections — M3). | Governed |
| `reject_uplift_factor` | 1.5 | 1.5 | `config.py` | **Tuning** | RI uplift. Active only when `run_ri_optimizer=false`; **overridden (inert)** when the optimizer is on. | Governed |
| `reject_max_risk_multiplier` | 3.0 | 3.0 | `config.py` | **Tuning** | RI per-bin cap. Same off/on duality as `reject_uplift_factor`. | Governed |
| `run_ri_optimizer` | `true` | `false` | `config.py` | **Tuning** | Auto-tunes the RI lever via Optuna. Toggle ≈ ±19%. When on, it **supersedes** the manual `reject_uplift`/`max_mult`. | Governed |
| `ri_calibration_gamma` | 0.8 | 1.0 | `config.py` | **Tuning** | The active RI lever **when the optimizer is on** (±24% on no_premium_cd in M3). | Governed |
| RI calibration basis | stress-free, no tasa_fin | — | `reject_inference_optimizer.py` | **Fixed** | Audit #29: the optimizer scores candidates on a stress-free, financing-unweighted blend (same raw-demand basis as the 1/a^γ target) and breaks calibration ties CONSERVATIVELY (lowest production in the 5% band). Pre-fix, the chosen uplift cancelled the stress factor (4.58→3.32 with stress on, −27%) and varied with tasa_fin. The target itself remains a modeling assumption — calibrated, not validated; M4 is the realized-outcome check. | Governed |
| `ri_optuna_n_trials` | 200 | 100 | `config.py` | **Tuning** | RI optimizer search budget. Affects runtime + tuning stability. | Governed |
| `mr_maturity_months` | 6 | 6 | `config.py` | **Expert** | **The maturity anchor.** Minimum months since booking for an account to count as H6-mature; younger accounts are excluded from realized risk to avoid diluting it with immature zeros. Used by the MR check (validation-only) and the M4 backtest. | Governed |
| `use_mr_outcomes` | `true` | `false` | `config.py` | **Expert** | Hybrid MR risk inference. **Validation/monitoring-only — does not feed the optimizer or the cutoffs (M3a).** | Governed |
| `cv_folds` | 4 | 4 | `config.py` | **Expert** | Cross-validation folds for model training (the #7 fresh-seed k-fold selection). | Governed || `selection_risk_basis` | `"point"` | `"point"` | `config.py` | **Expert** | Scenario selection rule. `"ci_upper"` (audit #28 Phase C) requires the candidate's bootstrap risk CI-upper ≤ target — materially more conservative (−61% production on no_premium_cd at the same 1.1% target). Enabling changes cutoffs ⇒ M5-style sign-off required. | Governed |

## Date windows
- **Main observation window:** `2024-06-01 → 2025-05-01` (`date_ini_book_obs` / `date_fin_book_obs`).
- **MR / out-of-time window:** `2025-06-01 → 2026-02-01` (`date_*_book_obs_mr`). Validated non-overlapping with the main
  window (a `UserWarning` fires on overlap, `config.py`). The M4 backtest auto-derives its mature held-out window from
  the data snapshot (`max(mis_date) − mr_maturity_months`).

## Model selection (audit #7)
The risk model is chosen by **fresh-seed k-fold CV with a working 1-SE rule** (simplest model within one standard error
of the best), with a winner-only 0.2 holdout report (never used for selection). On the production pooled `total` model
this selects **Linear Regression** (pre-#7 selected Lasso α≈0 — functionally identical). See the model validation report
and the #7 entry in `todo-list.md`.

## Reproducing the snapshot these values pin to
Each run writes `output/<segment>/data/run_lineage.json` (M2) capturing the data file **SHA-256**, mtime, row count, git
commit, config hash, and the headline assumptions. The committed golden reference
(`reports/validation/reference/<segment>_headline.json`) records the headline numbers **and** the SHA-256 they were
measured on. `run_reproducibility.py` re-derives the headline and fails loudly if the snapshot, code, or config drifted.

## Uncertainty of the optimized headline (audit #28)
The reported optimum is the **in-sample estimate of an optimized cutoff**: the selection picks the max-production
frontier point whose point-estimate risk sits under the target, so the headline risk is **optimistic at the binding
boundary (winner's curse)**. The **M4 out-of-time backtest is the unbiased check** of the frozen policy.

The bootstrap CI on the "Optimum selected" row (Phase A of #28):
- **Basis** — `risk_ci_*` is computed on the **blended booked + reject-inferred basis** (the same quantity as the
  headline `b2_ever_h6`), with the repesca component resampled via the rejected-population composition
  (`risk_ci_basis = "blended_booked_plus_ri"`). The booked-only realized CI is reported separately
  (`risk_booked_ci_*`). Before #28 the CI silently measured booked-only realized risk while decorating the blended
  headline, and the repesca production entered every replicate as a constant (zero variance).
- **What it captures** — sampling variance of the booked and score-rejected populations under the **fixed** chosen
  cutoff. **What it does not capture** — model/RI parameter uncertainty (the cell-level RI transformation is held
  fixed), and the re-optimization/selection step itself (quantified separately by the #28 Phase B selection-aware
  bootstrap and optimism estimate, when available).
