# Reproduction runbook (M5)

How an independent reviewer reproduces the headline numbers from the pinned inputs and confirms they are stable. No
prior knowledge of the codebase is assumed.

## 1. Environment
```bash
git clone <repo> && cd scoring_tools
git rev-parse HEAD                     # record the code revision under test
uv pip install -e .
uv run pytest tests/ -q                # sanity: the suite is green
```

## 2. Pin the data snapshot
The headline numbers are tied to one data file. Confirm you have the right snapshot:
```bash
shasum -a 256 data/demanda_direct_out.sas7bdat      # must match the reference's data_sha256
```
The expected SHA-256, mtime, row count, git commit, and config hash are recorded per run in
`output/<segment>/data/run_lineage.json` (M2 lineage) and in the committed golden reference
`reports/validation/reference/<segment>_headline.json`. **If the SHA-256 differs, you have a different snapshot** — the
headline must be re-validated (the reproducibility check will fail loudly on this).

## 3. Where the headline numbers live (per segment)
After a run, under `output/<segment>/data/`:
- `optimal_solution_base.csv` — chosen optimum: `b2_ever_h6` (risk %), `oa_amt_h0` (production €), `sol_fac`.
- `accepted_cells_base.csv` — the accepted score-bin coordinates (row count = accepted-cell count).
- `risk_production_summary_table_base.csv` — the "Optimum selected" row + **bootstrap CIs**
  (`risk_ci_lower/upper`, `production_ci_lower/upper`).
- `run_lineage.json` — the snapshot/code/config the numbers pin to.

## 4. Reproduce + check (the repeatable artifact)
```bash
# Re-derive the headline for a segment and compare to the committed golden reference (within tolerance):
uv run python run_reproducibility.py -s no_premium_cd \
  --config reports/validation/reference/no_premium_cd_config.toml

#  → PASS: headline reproduces within tolerance AND data SHA-256 matches the reference.
#  → FAIL: prints the reasons (risk/production drift, cell-count/accepted-set change, or DATA SNAPSHOT CHANGED).
```
> **Config for `no_premium_cd`.** The repo's root `config.toml` now targets the ecom portfolio (segments `new`/`known`),
> so it no longer contains `no_premium_cd`. The golden reference is pinned to the committed standalone config
> `reports/validation/reference/no_premium_cd_config.toml` (direct-channel data, `reject_inference_method="parceling"`,
> `use_mr_outcomes=true`) — pass it via `--config` for both the check and any re-establishment.
Tolerances default to risk ±0.01pp and production ±0.1%; override with `--risk-tol-pp` / `--prod-tol-pct`. A full
end-to-end re-run retrains the model (deterministic, seeded); pass `--model-path <dir>` to reuse a trained model and
reproduce only preprocessing→optimization (faster).

To (re)establish the reference from a trusted run (e.g. after an intentional, signed-off change):
```bash
uv run python run_reproducibility.py -s no_premium_cd \
  --config reports/validation/reference/no_premium_cd_config.toml --update-reference
```

> **Note — standalone vs production.** `run_reproducibility.py` reproduces a **standalone** run (`config.toml` +
> `segment_filter`), which trains the segment's own model. Production runs (`run_batch.py`) share one pooled `total`
> model across segments; the production headline is validated separately via the **out-of-time backtest**
> (`run_backtest.py`, M4) and the **#7 multi-segment validation** (both paths select near-identical linear models). The
> reproducibility check here proves the pipeline is *deterministic and reproducible from pinned inputs*.

## 5. Challenge the numbers (try to break them)
The validation report (`model_validation_report.md`) records an adversarial pass; an independent reviewer can re-run any
of these:
- **Out-of-time backtest** — `uv run python run_backtest.py -s <seg>` (realized vs predicted on a held-out cohort).
- **Config sensitivity** — `uv run python config_sensitivity.py -c <cfg>` (which knobs move the answer).
- **Bootstrap CIs** — already in `risk_production_summary_table_base.csv` (is the headline inside its CI; how wide).
- **Stability / drift** — `stability_psi_base.csv`, `drift_alerts_base.json` (PSI between main and MR periods).
- **Audit reconciliation** — `audit_base.csv` (loan-level sums reconcile to the summary KPIs).

## 6. Sign-off
Complete `mrm_signoff.md` (validator, model owner, date) once the reproduction PASSes and the residual risks are
accepted. Live / automated-cutoff use is gated on the conditions listed there.
