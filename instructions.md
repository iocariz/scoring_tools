# Instructions — Running the Scoring & Cutoff Optimization Pipeline

A practical runbook: how to set it up, how to run it, which knobs matter, how to tune for
better results, and the traps to avoid. For the full reference see `README.md`; this file is
the "just tell me how to use it" guide.

---

## 1. Setup

```bash
# Install (editable)
uv pip install -e .
# or: make setup

# Optional extras
uv pip install -e ".[ga]"        # pymoo — genetic-algorithm fallback for N>2 grids
uv sync --group dev               # dev tools: ruff, pytest, python-docx/-pptx
```

**You need two things before a real run:**

1. **The data file** — a SAS `.sas7bdat` export (default path `data/demanda_direct_out.sas7bdat`).
   Point at it with `data_path` in `config.toml`.
2. **Two config files** — `config.toml` (global defaults) and `segments.toml` (per-segment
   overrides). The segment config is recursively merged on top of the base config.

Sanity-check your install:

```bash
uv run pytest tests/ -q          # ~1300 tests on synthetic data, no real data needed
uv run python run_batch.py --list   # list the segments defined in segments.toml
```

---

## 2. Quick start (the happy path)

```bash
# 1. Run every segment, clean previous output, in parallel
uv run python run_batch.py --clean --parallel

# 2. Open the consolidated report
open output/consolidated_risk_production.html       # interactive dashboard
open output/consolidated_risk_production.xlsx       # management workbook + trust layer

# 3. (optional) Allocate a global risk budget across segments
uv run python run_allocation.py --target 1.0
```

`run_batch.py` is the normal entry point. It trains models (sharing them across supersegments),
optimizes each segment, **auto-runs the out-of-time backtest (M4)**, and writes the consolidated
report. A single segment can also be run standalone with `main.py` — but read the trap in §6 first.

---

## 3. The core workflow

```
config.toml + segments.toml          ← you edit these
        │
        ▼
run_batch.py  ──►  per-segment output/<segment>/   (cutoffs, report, CSVs, model)
        │          + output/backtest/              (M4 out-of-time backtest, automatic)
        │          + output/consolidated_*.{csv,html,xlsx}
        ▼
review the HTML / Excel report
        │
        ▼
validate (M4 backtest is automatic; add M5 reproducibility + policy registry — §7)
        │
        ▼
run_allocation.py   ← distribute a portfolio-wide risk budget across segments
```

1. **Configure** — set `variables`, date windows, `optimum_risk`, and the reject-inference /
   binning options (§5).
2. **Run the batch** — `run_batch.py [--parallel] [--clean]`.
3. **Review** — the per-segment `output/<segment>/report.html` and the consolidated workbook.
4. **Validate** — the M4 backtest runs automatically; pin golden numbers with
   `run_reproducibility.py` and track the live policy with `run_policy_registry.py` (§7).
5. **Allocate** (optional) — `run_allocation.py` picks one frontier point per segment to hit a
   global risk target.

---

## 4. How to run — the commands and the options that matter

### `run_batch.py` — multi-segment (use this by default)

```bash
uv run python run_batch.py [OPTIONS]
```

| Flag | Why you'd use it |
|:-----|:-----------------|
| `-s SEG [SEG ...]` | Run only specific segments |
| `--parallel` / `--workers N` | Run segments concurrently (faster) |
| `--clean` | Wipe previous output for the selected segments first |
| `--reuse-models` | Skip retraining supersegment models (much faster re-runs) |
| `--baseline` | Show the current booked portfolio as-is, **no optimization** (benchmark) |
| `--base-only` | Only the base scenario (skip pessimistic/optimistic) |
| `--allow-dq-warnings` | Proceed past **non-critical** data-quality warnings (see §6) |
| `--skip-dq-checks` | Skip data-quality checks entirely (not for production) |
| `--no-backtest` | Skip the automatic M4 out-of-time backtest |
| `--no-report` / `--no-consolidation` | Skip HTML reports / the consolidated report |
| `--consolidate-only` | Just regenerate the consolidated report from existing outputs |
| `--resimulate R [R ...]` | Re-run scenario analysis at new risk target(s) without re-training |
| `--log-file run.log` | Capture full DEBUG logs to a file |

### `main.py` — single segment

```bash
uv run python main.py [--config config.toml] [--training-only] [--baseline] [--base-only]
uv run python main.py --model-path output/_supersegment_x/models/model_...   # reuse a trained model
```

Use it for a focused single-segment run or to (re)train a supersegment model with
`--training-only`. **Note the standalone `segment_filter` trap in §6.**

### `run_allocation.py` — global risk budget across segments (after a batch)

```bash
uv run python run_allocation.py --target 1.0                  # one global risk target (%)
uv run python run_allocation.py --what-if 2.0,2.5,3.0         # compare several targets
uv run python run_allocation.py --target 1.0 --production-floor 50000
uv run python run_allocation.py --target 1.0 --lock no_premium_cd:3   # pin a segment's frontier point
```

The target is a **production-weighted** average of each segment's `b2_ever_h6`. Writes a policy
cutoff table + a plain-language narrative beside `--output`.

### Other runners

```bash
uv run python run_score_metrics.py            # Gini / lift / ROC / DeLong score discriminance
uv run python run_backtest.py                 # M4 backtest standalone (run_batch does it for you)
uv run python run_reproducibility.py -s SEG   # M5 golden-numbers check
uv run python run_policy_registry.py --compare  # champion vs challenger (§7)
uv run python dashboard.py                    # interactive results dashboard (localhost:8050)
uv run python analyze_logs.py run.log         # mine a log for config-tuning suggestions
```

---

## 5. Configuration for better results

All settings live under `[preprocessing]` in `config.toml`, override per segment in
`segments.toml`. Below are the choices that actually move the numbers.

### 5.1 The grid (`variables`, binning)

- **`variables`** — ≥ 2 score names; the optimization grid. Start with two core scores
  (internal × bureau), add a third (e.g. income) only if it discriminates risk.
- **Binning** — use **quantile** (equal-count, unsupervised) edges, the default and the only
  supported learned-edge method. Provide explicit `bin_edges` for established legacy tiers.
  > The legacy `"optimization"` (decision-tree) method is **deprecated** — it leaked the risk
  > target — and silently falls back to quantile. Don't use it.
- If a third dimension looks flat under quantile bins, it genuinely lacks signal — don't engineer
  around it with supervised splits.

### 5.2 Reject inference — **the single biggest lever** (`reject_*`)

Score-rejected applicants have no observed outcome but are exactly the population you're deciding
on. Correct for the selection bias:

```toml
reject_inference_method = "parceling"   # DEFAULT IS "none" — set this for any real run
reject_parceling_method = "linear"      # "linear" (default) | "power" | "sigmoid"
reject_uplift_factor    = 1.5           # 1.0–1.5 data-rich; 2.0–4.0 sparse/supersegment
reject_max_risk_multiplier = 3.0
stress_mode = "disabled"                # when parceling is ON, disable stress (see below)
```

- **Method shape:** `linear` (steady, the safe default), `power` (aggressive at low acceptance —
  heavy-tail risk), `sigmoid` (saturates — premium books with high acceptance).
- **Sparse segments:** turn on Bayesian smoothing so noisy per-bin rates shrink to the global rate:
  ```toml
  reject_bayesian_smoothing = true
  reject_bayesian_prior_strength = 10.0   # 10–50 typical; 100+ pulls hard to the global rate
  ```
- **Auto-tune** the uplift/cap on mature segments instead of guessing:
  ```toml
  run_ri_optimizer = true
  ri_optimizer_method = "grid"     # or "optuna"
  ri_calibration_gamma = 1.0       # lower (0.7/0.5) if it picks very aggressive params or MR degrades
  ```
  Keep `run_ri_optimizer = false` for supersegments (flat risk surface) and set manual factors.

### 5.3 Stress factor (`stress_mode`)

| Mode | When |
|:-----|:-----|
| `global` (default) | Conservative single scalar; fine when parceling is off |
| `per_bin` | Score variables span very different risk profiles |
| `disabled` | **Use this whenever parceling is active** — otherwise you double-count selection bias |

### 5.4 Out-of-time validation / H3→H6 (`*_mr*`)

Use real recent outcomes instead of pure model imputation:

```toml
date_ini_book_obs_mr = "2025-01-01"
date_fin_book_obs_mr = "2025-06-30"
use_mr_outcomes = true               # recommended
mr_extrapolation_method = "auto"     # fits the H3→H6 curvature from data — no manual tuning
mr_min_obs_per_bin = 30              # 50 for high volume, 10 for tiny segments
```

### 5.5 Risk appetite (`optimum_risk`, `risk_step`)

```toml
optimum_risk = 1.1     # target b2_ever_h6 in %, per business limits — set per segment
risk_step    = 0.1     # builds pessimistic (−step) and optimistic (+step) scenarios
```

Use `risk_step = 0.05` for sharp frontiers, `0.2+` to contrast starkly different strategies.

### 5.6 Constraints (committee guardrails)

```toml
max_swapin_production_pct = 10.0   # cap how much new production comes from untested (repesca) loans
max_swapin_risk           = 2.0    # hard cap on the swap-in population's own risk
```

Per-segment in `segments.toml`: `min_risk`, `max_risk`, `min_production`, `locked_sol_fac`
(for `run_allocation.py`), and `cutoff_floor_segment` for nested cutoffs across segments.

### 5.7 Don't touch these

- **`multiplier = 7`** and **`multiplier_h3 = 4`** are **fixed accounting constants** (H0…H6 /
  H0…H3 month counts). They are not tuning knobs — changing them silently rescales every risk
  number. Leave them.

---

## 6. Traps & troubleshooting

- **`main.py` on the default `config.toml` returns zero rows.** The default `segment_filter` is
  `"unknown"`, which matches nothing, so every row is filtered out. Fix: set a real
  `segment_filter` (a regex matching your data) in `config.toml`, **or** run via `run_batch.py`
  with `segments.toml` (which injects each segment's filter for you). Prefer `run_batch.py`.
- **The run halts on data-quality warnings.** DQ is **fail-closed by default** — both FAILED and
  WARNING tiers stop the run. Inspect the message first; if the warnings are acceptable, re-run
  with `--allow-dq-warnings` (relaxes only the WARNING tier; FAILED still halts). `--skip-dq-checks`
  bypasses DQ entirely (avoid for production).
- **Risk looks implausibly high after parceling.** You're probably double-counting selection bias:
  set `stress_mode = "disabled"` when `reject_inference_method = "parceling"`.
- **MILP returns infeasible.** Your constraints are too tight (aggressive `optimum_risk` + tight
  swap-in caps). Loosen a constraint; for N>2 grids install the GA extra (`.[ga]`) so the pymoo
  fallback can produce a near-optimal frontier.
- **Re-runs are slow.** Use `--reuse-models` to skip supersegment retraining, and `--resimulate`
  to re-score at new risk targets without re-training or re-optimizing.
- **Reproduce-ability / "did anything change?"** Every run writes
  `output/<segment>/data/run_lineage.json` (data SHA-256, git commit, config hash). Use it, plus
  `run_reproducibility.py`, to confirm a result still reproduces (§7).

---

## 7. Validation & governance (recommended before deploying cutoffs)

The pipeline ships a trust layer; use it before acting on a recommendation.

```bash
# M4 — out-of-time backtest (runs automatically inside run_batch; or standalone):
uv run python run_backtest.py -s no_premium_cd
#   → applies the FROZEN cutoffs to a matured held-out cohort; realized vs predicted risk with a
#     noise-aware flag (OK / INCONCLUSIVE / DRIFT). Surfaces on the Excel "Out-of-time Validation" sheet.

# M5 — pin & re-check the headline numbers (risk / production / accepted cells):
uv run python run_reproducibility.py -s no_premium_cd --update-reference   # establish the golden reference
uv run python run_reproducibility.py -s no_premium_cd                       # later: PASS iff numbers + data SHA still match

# Policy registry — track the live policy and compare a new one against it:
uv run python run_policy_registry.py --register -s no_premium_cd --make-champion   # record the live (champion) policy
uv run python run_policy_registry.py --compare  -s no_premium_cd                   # challenger (latest run) vs champion
#   → cell-level accept/reject diff + a noise-aware risk verdict (BETTER / WORSE / INCONCLUSIVE) on
#     the matured holdout. Registry committed under reports/policy_registry/<segment>.json.
```

The consolidated Excel (`consolidated_risk_production.xlsx`) collects this evidence on its
**Validation & Governance** and **Out-of-time Validation** sheets, plus bootstrap CI bands and a
"Recommendation & key risks" narrative on the Executive Summary. The validation pack
(`reports/validation/`) holds the assumptions register, reproduction runbook, model-validation
report, and the MRM sign-off.

---

## 8. Tuning cheat-sheet

| Goal | Set |
|:-----|:----|
| Correct selection bias (always) | `reject_inference_method = "parceling"`, `stress_mode = "disabled"` |
| Sparse segment / few obs per bin | `reject_bayesian_smoothing = true`, higher `reject_uplift_factor` (2–4), supersegment model |
| Mature, data-rich segment | `run_ri_optimizer = true` (let it pick uplift/cap) |
| Use real recent outcomes | `use_mr_outcomes = true`, `mr_extrapolation_method = "auto"` |
| Benchmark current policy (no optimization) | `--baseline` |
| Heavy-tail risk in rejected bins | `reject_parceling_method = "power"` |
| High-acceptance premium book | `reject_parceling_method = "sigmoid"` |
| Limit growth from untested loans | `max_swapin_production_pct`, `max_swapin_risk` |
| Faster iteration | `--reuse-models`, `--resimulate`, `--base-only` |
| Reproducibility/audit | `run_reproducibility.py`, `run_policy_registry.py`, `run_lineage.json` |

---

## 9. Where the outputs land

```
output/<segment>/
  report.html                         self-contained segment report
  data/                               cutoffs, frontier, MR comparison, stability, lineage, accepted_cells
  models/                             trained risk model + metadata + SHAP + cell-level CIs
output/
  consolidated_risk_production.{csv,html,xlsx}   portfolio rollup + management workbook (trust layer)
  backtest/                           M4 out-of-time backtest (auto)
  policy_registry/                    champion/challenger comparison (run_policy_registry --compare)
reports/
  validation/                         assumptions register, runbook, model-validation, MRM sign-off, golden references
  policy_registry/<segment>.json      committed registry of deployed policies (champion per segment)
```

---

**TL;DR:** edit `config.toml` + `segments.toml` → `uv run python run_batch.py --clean --parallel`
→ open `output/consolidated_risk_production.xlsx` → validate with the backtest / reproducibility /
policy-registry tools → `run_allocation.py` for the portfolio view. Always run with
`reject_inference_method = "parceling"` and `stress_mode = "disabled"`; never touch `multiplier`.
