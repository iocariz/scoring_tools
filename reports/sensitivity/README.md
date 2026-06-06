# M3 — config-sensitivity generality check

One-at-a-time (OAT) sweeps from `config_sensitivity.py` over the pipeline config knobs, ranking each by how much it
moves the headline outputs (accepted production %, portfolio risk, accepted-cell count). This round re-runs the
original `no_premium_cd` analysis with **widened Tier-1 grids** and adds a **second segment (`premium`)** to test
whether the tier classification generalizes.

> The intended second segment was `conso_known`, but it has **0 rows** in the current data snapshot (the
> `segment_cut_off` column has no such value), so `premium` (67,168 rows, a different client-quality tier) was
> substituted.

## What was run

| report CSV | segment | rows | optimizer | grids | baseline optimum |
|---|---|---|---|---|---|
| `premium_off_summary.csv` | premium | 67,168 | OFF | widened Tier-1, full registry | €26.96M / 1.0869% / 87 cells |
| `no_premium_cd_off_summary.csv` | no_premium_cd | 144,236 | OFF | widened Tier-1, full registry | €50.23M / 1.0899% / 129 cells |
| `premium_on_summary.csv` | premium | 67,168 | ON | scoped RI knobs, `ri_optuna_n_trials=30` | €26.96M / 1.0869% / 87 cells |

`*_detail.csv` holds the per-run production/risk/cells. Base config: `config.toml` (v7) with `segment_filter`
(+`run_ri_optimizer`) overridden; `multiplier=7`, `stress_mode="disabled"`, `reject_inference_method="parceling"`,
`use_mr_outcomes=true`.

### Methodology notes (honest)
- **Per-run full training.** `--reuse-model` was requested, but `train_shared_model` produced no reusable model dir and
  the harness **fell back to per-run training** — so every run retrained the risk model at the production Optuna trial
  counts. This *satisfies* "full Optuna trials" (stronger, not weaker) at the cost of ~3 h per OFF sweep; it also means
  the training-affecting `cv_folds` knob was swept rather than masked.
- **ON-mode scope.** A full-registry ON sweep at 200 RI trials/run is ~8 h, so the ON sweep was scoped to the RI-relevant
  knobs with `ri_optuna_n_trials` reduced to **30** (clearly a reduced-trials run). The RI optimizer genuinely executed
  (it selected `uplift=0.80, max_mult=1.00` on premium).
- Standalone per-segment models (not the pooled `total` model), matching how the original `no_premium_cd` sweep was run.

## Rankings (impact = max of production-% and risk-% spread)

**no_premium_cd OFF** — top movers: `multiplier` 52.3%, `stress_mode` 27.6%, `reject_uplift_factor` 13.3%,
`reject_parceling_method` 10.8%, then `cv_folds` 3.8% and a long tail ≤1.5%. Everything in the MR suite, `multiplier_h3`,
`monotonicity_*`, and `ri_calibration_gamma`/`run_ri_optimizer` (optimizer is OFF here) = **0.0**.

**premium OFF** — **only `multiplier` moves (55.6%)**; *every other knob is 0.0*, including `stress_mode`,
`reject_parceling_method`, `reject_uplift_factor`, `reject_max_risk_multiplier`.

**premium ON (scoped)** — all four RI knobs (`ri_calibration_gamma`, `reject_uplift_factor`,
`reject_max_risk_multiplier`, `reject_parceling_method`) = **0.0**.

## Findings

1. **`multiplier` dominates on both segments (52% / 56%) — but it is a FIXED accounting constant, not a tuning lever.**
   `todu_amt_pile_hN` is the *sum* of outstanding over months H0..HN, so `b2_ever_h6 = 7·Σ(todu_30ever_h6)/Σ(todu_amt_pile_h6)`
   uses `multiplier = 7` (H0..H6 inclusive = 7 months) to turn that sum into the average monthly outstanding;
   `multiplier_h3 = 4` (H0..H3). The widened grid's clean monotone response (premium €32.7M → €29.2M → €23.7M →
   €17.7M for 5 → 6 → 8 → 9) is therefore a **guardrail** result, not a degree of freedom: changing the constant
   mis-scales the risk denominator. Its large measured impact says **lock and validate it**, not "tune it". The real
   tunable space is everything below.

2. **The rest of the `no_premium_cd` Tier-1 set does NOT generalize.** `stress_mode`, `reject_uplift_factor`, and
   `reject_parceling_method` are Tier-1 on `no_premium_cd` but **byte-identical inert on premium**. The
   Tuning-tier materiality is **segment-dependent**, not a universal property of the pipeline.

3. **Root cause — reject-inference has nothing to bite on for premium.** Premium's rejections are **95.2% non-score**
   (only ~4.8% score-driven), vs **59.7% non-score** on `no_premium_cd`. With almost no score-rejection mass to parcel,
   every reject/RI knob is inert on premium; and stress is inert there too (the stress factor is ~constant across
   modes for this segment). The no_premium family has real score-rejection mass, so those levers bite.

4. **The "default-and-forget" (Expert/Tier-3) set generalizes — the reassuring result.** The MR suite, `multiplier_h3`,
   `monotonicity_relaxation`/`mono_z_threshold`, `pareto_n_points`, and `per_bin_tasa_fin` are ~0 on **both** segments.
   The M3 conclusion that the fragile/expert features are safe to leave at their (off) defaults **holds on a second
   segment**.

5. **RI mode-exclusivity confirmed mechanically.** In premium ON the RI optimizer ran and **overrode** the manual
   reject params (picked `uplift=0.80, max_mult=1.00`), which is exactly why `reject_uplift_factor`/
   `reject_max_risk_multiplier` are inert in ON mode — the optimizer supersedes them. `ri_calibration_gamma` is the
   active ON-mode lever where RI itself bites (`no_premium_cd`, prior partial sweep ±24%); on premium it is inert
   because RI is structurally inert there regardless of mode.

6. **Widened grids reproduce the original `no_premium_cd` ranking** (`multiplier` > `stress_mode` > `reject_uplift` >
   `reject_parceling`) — the prior finding is robust to grid resolution; widening only sharpened the magnitudes.

## Implication for M3 / governance

- **Defaults-are-safe holds and is strengthened** (finding 4): the expert/off-by-default knobs are inert on both
  segments tested.
- **Tuning-knob governance is segment-specific** (findings 2–3): `stress_mode` and the reject-inference knobs must be
  governed for the **no_premium / known-PL** family (where rejections are score-driven), but are effectively moot for
  **premium** (rejections are overwhelmingly non-score). The right control is *per-segment*, keyed on the
  score-rejection share — not one global tier list. This refines (does not contradict) the original classification.
- `multiplier`/`multiplier_h3` are **fixed metric-definition constants** (7 = H0..H6 months, 4 = H0..H3) — *lock and
  validate*, never tune. Their dominance in the sweep is a guardrail signal, not a governance lever.

This is a diagnostic/validation result — it changes no production defaults or pipeline behaviour and has no cutoff
impact.
