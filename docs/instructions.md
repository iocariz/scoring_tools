# Instructions — Running the Scoring & Cutoff Optimization Pipeline

A comprehensive, practical runbook: how to set it up, how it works, how to run every entry point,
every configuration knob (with defaults and ranges), how to tune for better results, how to read
the outputs, the traps to avoid, and the validation/governance workflow. `README.md` is the formal
reference; this file is the operational guide you keep open while working.

## Contents

1. [Setup](#1-setup)
2. [Quick start](#2-quick-start)
3. [How it works (concepts in 5 minutes)](#3-how-it-works-concepts-in-5-minutes)
4. [The core workflow](#4-the-core-workflow)
5. [The entry points and their options](#5-the-entry-points-and-their-options)
6. [Configuring `config.toml`](#6-configuring-configtoml)
7. [Configuring `segments.toml`](#7-configuring-segmentstoml)
8. [Tuning for better results](#8-tuning-for-better-results)
9. [Common workflows (step by step)](#9-common-workflows-step-by-step)
10. [Interpreting the outputs](#10-interpreting-the-outputs)
11. [Output files reference](#11-output-files-reference)
12. [Traps & troubleshooting](#12-traps--troubleshooting)
13. [Validation & governance](#13-validation--governance)
14. [Tuning cheat-sheet](#14-tuning-cheat-sheet)
15. [Glossary](#15-glossary)

---

## 1. Setup

```bash
# Install (editable)
uv pip install -e .                 # or: make setup
uv pip install -e ".[ga]"           # + pymoo, the genetic-algorithm fallback for N>2 grids
uv sync --group dev                 # dev tools: ruff, pytest, python-docx/-pptx

# Docker (reproducible environment)
docker build -t scoring-tools .     # or: make docker-build
docker run --rm -v "$PWD:/app" scoring-tools uv run python run_batch.py --list
```

**You need three things before a real run:**

1. **Data** — a SAS `.sas7bdat` export. Default path `data/demanda_direct_out.sas7bdat`; set
   `data_path` in `config.toml`. If the file has non-UTF-8 text, set `sas_encoding` (default
   `"latin-1"`).
2. **`config.toml`** — global defaults (§6).
3. **`segments.toml`** — per-segment overrides and supersegment definitions (§7). Recursively
   merged on top of `config.toml`.

Verify the install:

```bash
uv run pytest tests/ -q              # ~1300 tests on synthetic data (no real data needed)
uv run ruff check . && uv run ruff format --check .   # lint + format gate (what CI runs)
uv run python run_batch.py --list    # list the segments defined in segments.toml
```

---

## 2. Quick start

```bash
# Run every segment, clean previous output, in parallel; auto-runs the M4 backtest + consolidation
uv run python run_batch.py --clean --parallel

# Review
open output/consolidated_risk_production.html    # interactive portfolio dashboard
open output/consolidated_risk_production.xlsx     # management workbook + validation/governance trust layer
open output/<segment>/report.html                 # a single segment's self-contained report

# (optional) Distribute a global risk budget across segments
uv run python run_allocation.py --target 1.0
```

`run_batch.py` is the normal entry point. `main.py` runs a single segment but has a setup trap
(§12) — prefer the batch runner.

---

## 3. How it works (concepts in 5 minutes)

**Goal:** given applicants' credit scores, choose which score-grid cells to accept so booked
production is maximized while the annualized 6-month delinquency rate stays within an agreed risk
budget.

**Three populations** (by `reject_reason`):

| Population | Meaning | Role |
|:-----------|:--------|:-----|
| **Booked** | Approved & disbursed | Observed outcomes — used directly |
| **Score-rejected (repesca)** | Rejected by the score cutoff (`09-score`) | The population you're optimizing; outcomes **inferred** |
| **Policy-rejected** | Rejected for non-score reasons (`08-other`) | **Excluded** — rejection unrelated to the score |

**Risk metric** — `b2_ever_h6` (annualized, exposure-weighted 6-month "ever 30+ dpd" rate):

```
b2_ever_h6 = multiplier × todu_30ever_h6 / todu_amt_pile_h6      (multiplier = 7, a FIXED constant)
```

All flow indicators are annualized by `annual_coef = 12 / n_months` so periods of different length
are comparable. A complementary 3-month metric `b2_ever_h3` (`multiplier_h3 = 4`) is used for early
warning when the validation window hasn't fully matured.

**Repesca risk is inferred** through a sequential pipeline: trained risk model →
**stress factor** (uplift toward observed tail risk) → **reject-inference parceling** (uplift by
per-bin acceptance rate) → **`tasa_fin`** (transformation rate to booked-equivalent).

**Optimization** — each grid cell is a binary accept/reject variable. A MILP maximizes production
subject to (a) a linearized risk-budget constraint and (b) **monotonicity** (a "staircase": if a
safer cell is rejected, riskier cells must be too). The MILP is solved at ~50 risk targets to build
a **Pareto frontier** (max production per risk level). Three operating points are highlighted:

| Scenario | Risk target |
|:---------|:------------|
| Pessimistic | `optimum_risk − risk_step` |
| Base | `optimum_risk` |
| Optimistic | `optimum_risk + risk_step` |

**Validation** — selected cutoffs are checked out-of-time (the MR window, with H3→H6
extrapolation), with PSI/CSI stability, bootstrap CIs, and (post-run) the M4 backtest, M5
reproducibility, and the policy registry (§13).

---

## 4. The core workflow

```
config.toml + segments.toml          ← you edit these
        │
        ▼
run_batch.py  ──►  output/<segment>/          per-segment cutoffs, report, CSVs, model
        │          output/backtest/           M4 out-of-time backtest (automatic)
        │          output/consolidated_*.{csv,html,xlsx}
        ▼
review the HTML / Excel report (§10)
        │
        ▼
validate: M4 backtest (auto) + M5 reproducibility + policy registry (§13)
        │
        ▼
run_allocation.py    ← one frontier point per segment to hit a global risk target
```

---

## 5. The entry points and their options

### `run_batch.py` — multi-segment (default)

```bash
uv run python run_batch.py [OPTIONS]
```

| Flag | Description |
|:-----|:------------|
| `-s, --segments NAME …` | Run specific segments only |
| `-l, --list` | List configured segments and exit |
| `-p, --parallel` / `-w, --workers N` | Run segments concurrently (default workers = CPU count) |
| `-o, --output DIR` | Base output dir (default `output`) |
| `-c, --config PATH` / `--segments-config PATH` | Base / segments config files |
| `--reuse-models` | Reuse existing supersegment models (skip retraining) |
| `--clean` / `--clean-only` | Remove output dirs first / only clean |
| `--skip-dq-checks` | Skip data-quality checks entirely |
| `--allow-dq-warnings` | Proceed past **WARNING**-tier DQ (FAILED still halts) |
| `--no-consolidation` / `--consolidate-only` | Skip / only run the consolidated report |
| `--training-only` | Only DQ + model training |
| `--baseline` | Show current portfolio as-is, **no optimization** |
| `--base-only` | Only the base scenario |
| `--cutoff-ordering-mode {bottom_up,top_down}` | Sequential nested-cutoff direction |
| `--no-report` | Skip per-segment HTML reports |
| `--no-backtest` | Skip the automatic M4 out-of-time backtest |
| `--resimulate R …` | Re-run scenario analysis at new risk target(s) — no re-train/optimize |
| `--log-file PATH` | Capture full DEBUG logs to a file |

### `main.py` — single segment

```bash
uv run python main.py [OPTIONS]
```

| Flag | Description |
|:-----|:------------|
| `-c, --config PATH` | Config file (default `config.toml`) |
| `-m, --model-path DIR` | Reuse a pre-trained model (skip training) |
| `-t, --training-only` | Preprocessing + training only |
| `--baseline` / `--base-only` | Baseline mode / base scenario only |
| `--skip-dq-checks` / `--allow-dq-warnings` | DQ controls (see §12) |
| `--log-file PATH` | DEBUG logs to a file |
| `--resimulate R [R …]` | Reload cached artifacts, re-run scenarios at new risk target(s) |

> **Trap:** `main.py` on the default `config.toml` filters out *all* rows unless you set a real
> `segment_filter` (the default is `"unknown"`). See §12.

### `run_allocation.py` — global risk budget across segments (run after a batch)

```bash
uv run python run_allocation.py --target 1.0
uv run python run_allocation.py --what-if 2.0,2.5,3.0
```

| Flag | Description |
|:-----|:------------|
| `--target FLOAT` | Global risk target % (production-weighted average of segment `b2_ever_h6`) |
| `--what-if LIST` | Comma-separated extra targets — one optimization each + comparison pack |
| `--data-dir DIR` | Where to find `efficient_frontier_*.csv` (default `output`) |
| `--output PATH` | Primary CSV; its stem names the companion policy table + narrative |
| `--scenario NAME` | Scenario to use (default `base`) |
| `--method {exact,greedy}` | MILP (exact, default) or hill-climbing (greedy) |
| `--segments-config PATH` | For per-segment `min_risk`/`max_risk`/`min_production` |
| `--production-floor FLOAT` | Global minimum production constraint |
| `--lock SEGMENT:SOL_FAC` | Pin a segment to a frontier point (repeatable) |

### Validation / governance runners

```bash
uv run python run_backtest.py [-s SEG …]            # M4: frozen cutoffs on a matured holdout (auto in batch)
uv run python run_reproducibility.py -s SEG          # M5: golden-numbers check
uv run python run_policy_registry.py --compare       # champion vs challenger (§13)
```

`run_reproducibility.py` flags: `-s/--segment` (required), `-c/--config`, `--reference`,
`-o/--output`, `--scenario`, `--risk-tol-pp` (0.01), `--prod-tol-pct` (0.1), `--model-path`,
`--update-reference`.
`run_backtest.py` flags: `-s/--segments`, `-c/--config`, `--data-dir`, `-o/--output`,
`--scenario`, `--maturity-months` (6), `--holdout-start`, `--holdout-end`.
`run_policy_registry.py` flags: `--register | --list | --compare` (one required), `-s/--segments`,
`--make-champion`, `--data-dir`, `-o/--output`, `--registry-dir`, `--scenario`,
`--maturity-months`, `--holdout-start`, `--holdout-end`.

### Analysis & UI

```bash
uv run python run_score_metrics.py                   # Gini / lift / ROC / DeLong score discriminance
uv run python run_selection_bias_analysis.py         # Thorndike correction, reject-inference Gini
uv run python dashboard.py [-p 8050]                 # interactive results dashboard (localhost)
uv run python interactive_allocator.py [-p 8051]     # interactive global allocation
uv run python generate_presentation.py [--pdf]       # .pptx / .pdf deck
uv run python analyze_logs.py run.log                # mine a log for config-tuning suggestions
```

> Dashboards bind to loopback by default. Binding to a non-localhost host requires
> `DASHBOARD_AUTH_USER` / `DASHBOARD_AUTH_PASS` (HTTP Basic Auth).

Makefile shortcuts: `make run`, `make run-batch`, `make test`, `make lint`, `make format`,
`make clean`, `make setup`, `make docker-build`, `make docker-run`.

---

## 6. Configuring `config.toml`

All settings live under `[preprocessing]`. Below is an annotated template, then the full grouped
reference.

```toml
[preprocessing]
# --- Data & segment ---
data_path     = "data/demanda_direct_out.sas7bdat"
sas_encoding  = "latin-1"
segment_filter = "unknown"          # regex; OVERRIDE per segment in segments.toml (default matches nothing)
keep_vars   = ["mis_date", "reject_reason", "status_name", "..."]   # columns to retain (dataset-specific)
indicators  = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0", "..."]  # target + amount columns
log_level   = "INFO"

# --- Grid variables (>= 2) and model training subset ---
variables           = ["sc_octroi_new_clus", "new_efx_clus"]   # the optimization grid
inference_variables = ["sc_octroi_new_clus", "new_efx_clus"]   # subset used to TRAIN the model (default = variables)

# --- Observation window (main period) ---
date_ini_book_obs = "2024-01-01"
date_fin_book_obs = "2024-12-31"
n_months = 12                       # rolling window for the transformation rate

# --- Economics (multiplier/multiplier_h3 are FIXED accounting constants — do not tune) ---
multiplier    = 7.0
multiplier_h3 = 4.0
optimum_risk  = 1.1                 # target b2_ever_h6 % — set per segment
risk_step     = 0.1                 # pessimistic = optimum-step, optimistic = optimum+step

# --- Binning (learned quantile edges; explicit bin_edges for legacy tiers) ---
[preprocessing.bins.income_bin]
source_col = "income_t1t2_m"
output_col = "income_bin"
max_bins   = 3                       # learn equal-count edges; OR set bin_edges = [-inf, 1500.0, inf]
method     = "quantile"             # "optimization" is DEPRECATED (target leakage) — falls back to quantile

[preprocessing.directions]
sc_octroi_new_clus = -1              # -1: higher bin = safer | 1: higher bin = riskier
new_efx_clus       = -1

# --- Reject inference (the biggest lever) ---
reject_inference_method  = "parceling"   # DEFAULT IS "none" — set this for a real run
reject_parceling_method  = "linear"      # linear | power | sigmoid
reject_uplift_factor     = 1.5
reject_max_risk_multiplier = 3.0
stress_mode = "disabled"             # disable stress whenever parceling is ON (avoids double-counting)

# --- Out-of-time validation / H3->H6 ---
date_ini_book_obs_mr = "2025-01-01"
date_fin_book_obs_mr = "2025-06-30"
use_mr_outcomes         = true
mr_extrapolation_method = "auto"
mr_min_obs_per_bin      = 30
```

### Full parameter reference (defaults / ranges)

**Core (required):** `keep_vars`, `indicators`, `variables` (≥2, unique), `date_ini_book_obs`,
`date_fin_book_obs`.

| Group | Key | Default | Range / values | Notes |
|:------|:----|:--------|:---------------|:------|
| Data | `data_path` | `data/demanda_direct_out.sas7bdat` | path | SAS source |
| Data | `sas_encoding` | `"latin-1"` | | text encoding |
| Data | `segment_filter` | `"unknown"` | regex | override per segment |
| Data | `inference_variables` | `None`→`variables` | subset of `variables` | train on fewer dims than the grid |
| Data | `score_measures` | `None` | list | for `run_score_metrics.py` |
| Data | `log_level` | `"INFO"` | | |
| Economic | `multiplier` | `7.0` | >0 | **FIXED** (H0..H6) — don't tune |
| Economic | `multiplier_h3` | `4.0` | >0 | **FIXED** (H0..H3) — don't tune |
| Economic | `optimum_risk` | `1.1` | (0,100] | target risk %, per segment |
| Economic | `risk_step` | `0.1` | (0,50] | scenario spread |
| Economic | `n_months` | `12` | | transformation-rate window |
| Economic | `z_threshold` | `3.0` | >0 | outlier Z-score |
| Binning | `bins.<v>.method` | `"quantile"` | quantile / ~~optimization~~ | optimization deprecated |
| Binning | `bins.<v>.max_bins` / `bin_edges` | — | | one of the two required |
| Binning | `directions.<v>` | auto | `-1` safer-up / `1` riskier-up | monotonicity |
| Model | `cv_folds` | `4` | 2–10 | |
| Model | `model_hurdle_per_loan` | `false` | | offer per-loan HurdleRegressor candidate |
| Stress | `stress_mode` | `"global"` | global / per_bin / disabled | use `disabled` with parceling |
| tasa_fin | `per_bin_tasa_fin` | `false` | | per-cell transformation rate |
| Reject inf. | `reject_inference_method` | `"none"` | none / parceling | **set `parceling`** |
| Reject inf. | `reject_parceling_method` | `"linear"` | linear / power / sigmoid | |
| Reject inf. | `reject_uplift_factor` | `1.5` | 0–10 | 1–1.5 rich, 2–4 sparse |
| Reject inf. | `reject_max_risk_multiplier` | `3.0` | 1–10 | cap |
| Reject inf. | `reject_bayesian_smoothing` | `false` | | sparse segments |
| Reject inf. | `reject_bayesian_prior_strength` | `10.0` | (0,1000] | 10–50 typical |
| Reject inf. | `reject_enforce_monotonicity` | `false` | | isotonic on multipliers |
| Reject inf. | `reject_no_demand_anchor_percentile` | `0.10` | 0–0.5 | conservative low anchor |
| Reject inf. | `reject_confidence_scale` | `10.0` | (0,1000] | shrinkage count scale |
| Reject inf. | `reject_acceptance_recent_months` | `None` | ≥1 | recent-window rates |
| Reject inf. | `reject_acceptance_decay_half_life_months` | `None` | >0 | time-decayed rates |
| Reject inf. | `reject_apply_h3_multiplier` | `false` | | also uplift H3 numerator |
| Reject inf. | `reject_include_all_rejections` | `false` | | **deprecated & ignored** |
| RI optimizer | `run_ri_optimizer` | `false` | | auto-tune uplift/cap |
| RI optimizer | `ri_optimizer_method` | `"grid"` | grid / optuna | |
| RI optimizer | `ri_calibration_gamma` | `1.0` | (0,1] | lower = less aggressive |
| RI optimizer | `ri_validation_split` | `0.7` | (0,1] | holdout for OOT validation |
| RI optimizer | `ri_uplift_range` / `ri_max_mult_range` | `[0,5]` / `[1,5]` | | search ranges |
| RI optimizer | `ri_uplift_steps` / `ri_max_mult_steps` | `11` / `9` | | grid = 99 combos |
| RI optimizer | `ri_optuna_n_trials` | `100` | 10–10000 | |
| MR | `date_ini_book_obs_mr` / `_fin_` | `None` | | both required if either set |
| MR | `use_mr_outcomes` | `false` | | enable H3→H6 inference |
| MR | `mr_min_obs_per_bin` | `30` | ≥1 | 50 high-vol, 10 tiny |
| MR | `mr_maturity_months` | `6` | 0–24 | H6 maturity filter |
| MR | `mr_extrapolation_method` | `"linear"` | linear/power/logistic/auto | **`auto` recommended** |
| MR | `mr_extrapolation_curvature` | `1.0` | 0.3–5 | power exponent (ignored for auto) |
| MR | `mr_extrapolation_risk_multiplier` | `3.0` | (0,10] | relative cap |
| MR | `mr_extrapolation_hard_cap` | `15.0` | (0,100] | absolute cap % |
| MILP | `milp_time_limit` | `30.0` | >0 | seconds |
| MILP | `pareto_n_points` | `50` | 5–500 | sweep density |
| MILP | `n_bootstraps` | `1000` | 100–50000 | CI replicates |
| MILP | `monotonicity_relaxation_enabled` | `false` | | relax sparse/ambiguous adjacencies |
| MILP | `monotonicity_uncertainty_min_exposure` | `0.0` | ≥0 | relax only **below** this exposure |
| MILP | `monotonicity_uncertainty_z_threshold` | `1.0` | ≥0 | ambiguity threshold |
| Swap-in | `max_swapin_production_pct` | `None` | 0–100 | cap repesca production share |
| Swap-in | `max_swapin_risk` | `None` | 0–100 | cap repesca own risk |
| Sensitivity | `run_sensitivity` | `false` | | flip-threshold analysis |
| Mode | `baseline_mode` | `false` | | no optimization |
| Mode | `base_scenario_only` | `false` | | base scenario only |
| DQ | `dq_allow_warnings` | `false` | | fail-closed by default |
| Ordering | `cutoff_floor_segment` | `None` | (per segment) | nested cutoffs |

---

## 7. Configuring `segments.toml`

```toml
# --- Supersegments: share a trained model and/or group in the report ---
[modelling_supersegments.total]
segment_filters = ["np_ab", "np_cd", "np_ef"]      # train ONE model on the combined population

[reporting_supersegments.pl_new]
segment_filters = ["np_ab", "np_cd"]
bin_edges.income_bin = [-inf, 1500.0, inf]          # fixed edges for this report group
# learn_own_bin_edges = true                        # OR learn edges from this group's own population

# --- Segments ---
[segments.np_cd]
segment_filter         = "no_premium_cd"           # (required) regex matching the data
modelling_supersegment = "total"                    # shared model
reporting_supersegment = "pl_new"                   # report grouping
optimum_risk = 1.0
risk_step    = 0.05
reject_parceling_method = "power"                    # any config.toml field can be overridden here
stress_mode  = "disabled"

# Allocation constraints (used by run_allocation.py)
min_risk = 0.5
max_risk = 1.5
min_production = 1_000_000
# locked_sol_fac = 3                                 # pin to a specific frontier point

# Sequential nested cutoffs: this segment must accept a SUPERSET of np_ef's cells
cutoff_floor_segment = "np_ef"

# Force-reject cells below a per-variable threshold (scalar or income_bin-keyed map)
min_accepted_bin_by_variable = { new_efx_clus = 4, sc_octroi_new_clus = 7 }

[segments.np_cd.directions]
sc_octroi_new_clus = -1
new_efx_clus       = -1
```

Resolution: `modelling_supersegment > supersegment > None`,
`reporting_supersegment > supersegment > None`. A plain `[supersegments.NAME]` sets both.

---

## 8. Tuning for better results

**Always:**

- `reject_inference_method = "parceling"` (the default `"none"` does no selection-bias correction).
- `stress_mode = "disabled"` whenever parceling is active (otherwise you double-count the bias).
- **Quantile** binning (the `"optimization"` method is deprecated/leaky and falls back to quantile).
- Never touch `multiplier` / `multiplier_h3` — they are fixed accounting constants.

**Parceling method by book shape:**

| Method | Best for |
|:-------|:---------|
| `linear` (default) | General use; steady, interpretable penalty |
| `power` | Heavy-tail risk concentrated in highly-rejected bins (subprime) |
| `sigmoid` | High-acceptance premium books where risk saturates |

**Sparse segments / few observations per bin:**

```toml
reject_bayesian_smoothing = true
reject_bayesian_prior_strength = 10.0   # 10–50; 100+ pulls hard toward the global rate
reject_uplift_factor = 3.0              # aggressive manual uplift
run_ri_optimizer = false                # flat surface → don't auto-tune
modelling_supersegment = "total"        # borrow a shared model
```

**Mature, data-rich segments:** let the optimizer pick uplift/cap:

```toml
run_ri_optimizer = true
ri_optimizer_method = "optuna"          # or "grid"
ri_calibration_gamma = 1.0              # drop to 0.7/0.5 if it picks very aggressive params or MR degrades >2x
```

**Out-of-time realism:** `use_mr_outcomes = true` + `mr_extrapolation_method = "auto"` (fits the
H3→H6 curvature from data — no manual tuning).

**Committee guardrails:** `max_swapin_production_pct`, `max_swapin_risk`, per-segment
`min_risk`/`max_risk`/`min_production`, and `cutoff_floor_segment` for nested cutoffs.

---

## 9. Common workflows (step by step)

### Add a new segment

```toml
# segments.toml
[segments.my_seg]
segment_filter = "my_regex"
optimum_risk = 1.2
reject_inference_method = "parceling"
stress_mode = "disabled"
```
```bash
uv run python run_batch.py -s my_seg --clean
```

### Fixed (committee-approved) cutoffs — skip optimization

```toml
[segments.my_seg.fixed_cutoffs]
sc_octroi_new_clus = [1.0, 2.0, 3.0, 4.0]   # 2-var: paired bins/cutoffs (equal length)
new_efx_clus       = [3, 4, 5, 6]
strict_validation  = true                    # error (not warn) on off-grid cutoffs
run_all_scenarios  = true                    # also emit pessimistic/optimistic
```

### Sequential nested cutoffs across segments

Set `cutoff_floor_segment` per segment (`ef` ⊆ `cd` ⊆ `ab`), then:
```bash
uv run python run_batch.py --cutoff-ordering-mode bottom_up   # tightest first (floor constraints)
```

### Resimulate at new risk targets (no re-train/optimize)

```bash
uv run python run_batch.py --resimulate 0.8 1.2 1.6
uv run python run_batch.py --resimulate scenarios.toml        # per-segment targets
```

### Benchmark the current policy (no optimization)

```bash
uv run python run_batch.py --baseline      # Optimum = Actual; MR inference still runs
```

### Global allocation what-if

```bash
uv run python run_allocation.py --what-if 1.0,1.5,2.0 --output allocation.csv
# → allocation_what_if.csv + allocation_allocation_narratives.md
```

### Promote a new policy (governance loop)

```bash
uv run python run_batch.py -s my_seg                                  # produce a fresh policy
uv run python run_policy_registry.py --register -s my_seg --make-champion   # first time: record the live policy
# next cycle:
uv run python run_policy_registry.py --compare -s my_seg              # challenger vs champion on matured holdout
# if the verdict is good, promote:
uv run python run_policy_registry.py --register -s my_seg --make-champion
```

---

## 10. Interpreting the outputs

**Risk–production summary (per scenario, main & MR).** Four rows:

| Row | Meaning |
|:----|:--------|
| **Actual** (booked) | Current policy baseline |
| **Swap-in** (repesca accepted) | Upside from opening cutoffs |
| **Swap-out** (booked rejected) | Downside from tightening cutoffs |
| **Optimum** | Net result under the proposed cutoffs |

**Scenarios** — pessimistic / base / optimistic bracket the recommendation across risk appetites.
Read the base scenario as the recommendation; the others quantify the cost/benefit of moving the
risk target by `risk_step`.

**Bootstrap CIs** — every headline risk/production carries a 95% CI (1,000 resamples). A wide band
means the estimate rests on few defaults — treat point differences inside the band as noise.

**M4 backtest flag** (Out-of-time Validation sheet): `OK` (CIs overlap — within noise),
`INCONCLUSIVE` (too few OOT defaults), `DRIFT` (≥10 defaults **and** OOT CI sits entirely above the
in-sample CI). Only `DRIFT` is an action signal.

**Policy-registry verdict** (champion vs challenger): `BETTER`/`WORSE` only when both policies have
≥10 realized defaults **and** their risk CIs are fully separated; else `INCONCLUSIVE`. The verdict
is risk-only — read the production delta beside it.

**PSI / CSI stability:** `< 0.1` stable, `0.1–0.25` moderate (investigate), `≥ 0.25` unstable
(hold/refresh).

**`risk_source` column** (MR comparison): which evidence drove each bin — `mr_observed` (direct
mature H6), `h3_extrapolated`, `main_imputed`, or `model_fallback`.

---

## 11. Output files reference

### Per-segment — `output/<segment>/`

| File | Contents |
|:-----|:---------|
| `report.html` | Self-contained segment report (cutoffs, MR, PSI, sensitivity, lineage) |
| `data/optimal_solution_{scenario}.csv` | Selected cutoffs + KPIs |
| `data/accepted_cells_{scenario}.csv` | The accepted-cell allow-list (base) — the policy primitive |
| `data/pareto_optimal_solutions.csv` / `efficient_frontier_{scenario}.csv` | Pareto frontier |
| `data/risk_production_summary_table_{scenario}.csv` | Actual/Swap-in/Swap-out/Optimum |
| `data/cutoff_summary_wide.csv` | Per-cell accept/reject + per-cell CIs |
| `data/mr_risk_comparison_{scenario}.csv` | Per-bin MR drift + `risk_source` |
| `data/stability_psi_{scenario}.csv` / `drift_alerts_{scenario}.json` | PSI/CSI + alerts |
| `data/sensitivity_*.csv` / `cell_marginal_impact_*.csv` | Sensitivity & marginal impact (if enabled) |
| `data/ri_optimizer_results.csv` | RI parameter search (if `run_ri_optimizer`) |
| `data/run_lineage.json` | Provenance: data SHA-256, git, config hash, assumptions |
| `models/model_*/` | Trained model + metadata + SHAP + cell-level CIs |

### Consolidated & governance

| Path | Contents |
|:-----|:---------|
| `output/consolidated_risk_production.{csv,html,xlsx}` | Portfolio rollup + management workbook (trust layer) |
| `output/backtest/` | M4 backtest (auto): `backtest_consolidated_*.csv`, per-segment, `_calibration`, summary |
| `output/policy_registry/` | Champion/challenger comparison (`run_policy_registry --compare`) |
| `output/_supersegment_<name>/` | Shared model artifacts |
| `reports/policy_registry/<segment>.json` | Committed registry (champion per segment) |
| `reports/validation/` | Assumptions register, reproduction runbook, model-validation report, MRM sign-off, golden references |

---

## 12. Traps & troubleshooting

- **`main.py` returns zero rows.** Default `segment_filter = "unknown"` matches nothing. Fix: set a
  real `segment_filter` in `config.toml`, or run via `run_batch.py` + `segments.toml` (injects it).
  Prefer `run_batch.py`.
- **Run halts on data-quality warnings.** DQ is **fail-closed** — both FAILED and WARNING tiers
  stop the run. Inspect first; re-run with `--allow-dq-warnings` to relax only WARNINGs, or
  `--skip-dq-checks` to bypass entirely (not for production).
- **Risk implausibly high after parceling.** Double-counted bias — set `stress_mode = "disabled"`.
- **MILP infeasible.** Constraints too tight (aggressive `optimum_risk` + tight swap-in caps).
  Loosen a constraint; for N>2 install `.[ga]` so the pymoo fallback can run.
- **Re-runs slow.** `--reuse-models` (skip retraining), `--resimulate` (re-score only),
  `--base-only`, `-s` to scope segments.
- **Income/3rd dimension looks flat.** Under quantile bins that means no signal — don't reach for
  the deprecated supervised `"optimization"` method (it leaks the target).
- **"Did anything change?"** Compare `run_lineage.json` data SHA-256 across runs;
  `run_reproducibility.py` fails loudly if the snapshot changed.

---

## 13. Validation & governance

Run before acting on a recommendation:

```bash
# M4 — out-of-time backtest (automatic in run_batch; or standalone):
uv run python run_backtest.py -s my_seg
#   frozen cutoffs on a matured held-out cohort; realized vs predicted risk + noise-aware flag.

# M5 — pin and re-check the headline numbers:
uv run python run_reproducibility.py -s my_seg --update-reference    # establish the golden reference
uv run python run_reproducibility.py -s my_seg                        # PASS iff numbers + data SHA still match

# Policy registry — track the live policy and compare a candidate:
uv run python run_policy_registry.py --register -s my_seg --make-champion
uv run python run_policy_registry.py --compare  -s my_seg
```

The consolidated Excel collects this on its **Validation & Governance** and **Out-of-time
Validation** sheets, with bootstrap CI bands and a "Recommendation & key risks" narrative on the
Executive Summary. The committed validation pack lives in `reports/validation/`.

---

## 14. Tuning cheat-sheet

| Goal | Set |
|:-----|:----|
| Correct selection bias (always) | `reject_inference_method = "parceling"`, `stress_mode = "disabled"` |
| Sparse segment | `reject_bayesian_smoothing = true`, `reject_uplift_factor` 2–4, supersegment model |
| Mature, data-rich segment | `run_ri_optimizer = true` |
| Use real recent outcomes | `use_mr_outcomes = true`, `mr_extrapolation_method = "auto"` |
| Heavy-tail rejected risk | `reject_parceling_method = "power"` |
| High-acceptance premium book | `reject_parceling_method = "sigmoid"` |
| Benchmark current policy | `--baseline` |
| Limit growth from untested loans | `max_swapin_production_pct`, `max_swapin_risk` |
| Faster iteration | `--reuse-models`, `--resimulate`, `--base-only`, `-s` |
| Reproducibility / audit | `run_reproducibility.py`, `run_policy_registry.py`, `run_lineage.json` |

---

## 15. Glossary

| Term | Meaning |
|:-----|:--------|
| **b2_ever_h6 / _h3** | Annualized exposure-weighted 6- / 3-month ever-30+dpd rate (the risk metric) |
| **Repesca** | Score-rejected (`09-score`) applicants — the population being optimized |
| **Swap-in / swap-out** | Repesca newly accepted / booked newly rejected under a proposed policy |
| **Stress factor** | Uplift of repesca risk toward the observed tail risk of the booked population |
| **Parceling** | Reject-inference uplift of risk by each bin's acceptance rate |
| **tasa_fin** | Transformation rate — fraction of eligible applications ultimately disbursed |
| **Cell / mask** | One grid coordinate / the binary accept-reject vector over all cells |
| **Pareto frontier** | The set of policies where no other has both lower risk and higher production |
| **MR period** | Recent-Monitoring out-of-time window used to validate, not optimize |
| **Champion / challenger** | The live policy / a candidate scored against it on a common cohort |
| **Supersegment** | Segments sharing a trained model (modelling) and/or report group (reporting) |
| **Monotonicity (staircase)** | If a safer cell is rejected, riskier cells along that axis must be too |

---

**TL;DR:** edit `config.toml` + `segments.toml` → `uv run python run_batch.py --clean --parallel`
→ open `output/consolidated_risk_production.xlsx` → validate with the backtest / reproducibility /
policy-registry tools → `run_allocation.py` for the portfolio view. Always run with
`reject_inference_method = "parceling"` and `stress_mode = "disabled"`; never touch `multiplier`.
