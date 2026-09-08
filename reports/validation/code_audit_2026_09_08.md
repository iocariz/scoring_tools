# Code audit — 2026-09-08

Six confirmed defects: three P1 findings to address before relying on affected optimization or promotion results, and three P2 reliability findings. No production code was changed.

Reviewed the working tree at commit `4ca1bf6f63e7d23269ec9444c88b20ded1619414`, including the existing uncommitted V2 implementation. Context: `CLAUDE.md`, `README.md`, `todo-list.md`, and repository guidance. This was a source and synthetic-behavior audit of preprocessing, risk fitting, uncertainty, optimization, model reuse, policy registration/comparison, and V2 accounting/fitting/optimization. It is not an exhaustive proof of correctness or a fresh real-data validation.

## F1 — P1: Incomplete booked H6 outcomes enter the optimizer as zero defaults

> **Status: FIXED (2026-09-08).** One joint completeness rule (`mask_incomplete_outcome_pairs`, `src/utils.py`) now applies at the optimizer aggregation, the fixed-mask bootstrap, and the selection-aware bootstrap: a loan with a partially realized H6 (or H3) outcome contributes to NEITHER the numerator nor the denominator of that pair, while production still counts it — the same rule the M4 backtest already used. HRI pairs are exempt (sparse by design). The F1 reproduction below now asserts the corrected behavior. Independent verification note: the current direct extract has zero partially-realized booked H6 rows, so headline numbers are unchanged by this fix.

Location: [src/inference_optimized.py:2094](/Users/inigo_ocariz_laptop/src/scoring_tools/src/inference_optimized.py:2094).

**Trigger.** A booked loan has a missing H6 numerator and a populated exposure denominator.

**Verified behavior.** The real aggregation and MILP accept €1,000 at 0.7% risk under a 1% cap. The same policy's complete booked observations give 7.0%; one row is incomplete. The risk of the full portfolio is unknown.

**Cause and consequence.** The booked aggregation sums each indicator independently with NaN-skipping semantics. The missing numerator contributes zero while its exposure and production remain. Training and M4 use different completeness rules, and DQ exempts these fields from missing-value checks.

**Suggested correction.** Define one booked-H6 completeness rule before aggregation. Retain production accounting separately and fail, exclude the affected cells, or explicitly estimate unknown risk; do not admit it as observed zero. Apply the rule to selection and fixed-policy bootstraps.

**Prior audit context.** New extension of TODO #41: the backtest fix is present, but production aggregation is still affected.

## F2 — P1: Zero exposure defeats the champion/challenger observability guard

> **Status: FIXED (2026-09-08).** The observability mask now requires positive, finite H6 exposure (`todu_amt_pile_h6 > 0`), not merely non-null fields — a 0/0 booked row no longer counts as an observed outcome. The reproduction below asserts the corrected behavior: the added cell is unobservable, its demand share is preserved, and the BETTER verdict is blocked (INCONCLUSIVE with the survivorship message). Trigger prevalence at fix time: 14,642 booked 0/0 rows (13%) in the current direct extract — this finding was live.

Location: [src/policy_registry.py:475](/Users/inigo_ocariz_laptop/src/scoring_tools/src/policy_registry.py:475).

**Trigger.** An added cell has a booked row with H6 numerator=0 and denominator=0, but no positive H6 exposure.

**Verified behavior.** The comparison returns BETTER and unobservable share=0, although 90.92% of challenger-accepted demand lies in that added cell. Both policies clear the default-count gate.

**Cause and consequence.** The guard treats both fields being non-null as an observed outcome. A 0/0 rate is undefined, so this row supplies no evidence about the added cell's risk.

**Suggested correction.** Require positive, finite H6 exposure when establishing whether an added cell is observed. Preserve the unobservable-demand share and block BETTER when it exceeds the threshold.

**Prior audit context.** Residual in TODO #31's observability guard; unrelated to the already-fixed paired-bootstrap calculation.

## F3 — P1: Reusing an older risk model loads the latest exposure model

> **Status: FIXED (2026-09-08).** New trainings persist the exposure model (+ SHA-256 sidecar) inside the versioned `model_<timestamp>/` directory — the verified pair, resolved first at load time. Legacy directories without an in-dir copy may still use the shared root copy ONLY when the selected directory is the newest `model_*` there (that same run wrote the root copy), with a warning; selecting an OLDER directory fails loudly instead of silently pairing with the newest exposure model. The reproduction below asserts all three behaviors.

Location: [src/pipeline/inference.py:106](/Users/inigo_ocariz_laptop/src/scoring_tools/src/pipeline/inference.py:106).

**Trigger.** Train twice into the same models directory, then use --model-path to select the older model_<timestamp> directory.

**Verified behavior.** The loader resolves the unversioned sibling todu_model.joblib. The path-resolution reproduction requests model_old and obtains the later exposure coefficient 2 instead of the original 7. Deserialization is mocked; the resolver is real.

**Cause and consequence.** Risk models are versioned, while the exposure model is overwritten at the common models root on every training run. Integrity checks validate the individual file, not its pairing with the selected risk model.

**Suggested correction.** Save both fitted models and a manifest inside the same versioned directory. Reuse and validate that exact pair.

**Prior audit context.** Additional model-reuse failure mode beyond the bin-edge checks added for TODO #40.

## F4 — P2: A changed grid can register under the old policy ID

> **Status: FIXED (2026-09-08).** The policy id now carries a grid fingerprint (`grid_fingerprint`): ordered axis names, each axis's raw source column, its frozen cutpoints, and the accepted-cell set — genuinely different score regions get different ids, so `register_policy`'s idempotent no-op can no longer swallow a changed grid nor promote a stale entry. `PolicyEntry` also persists `bin_sources` (var → raw source column) for transparency, partially closing F5's registry-side gap; legacy entries without the field still load. The reproduction below asserts the corrected behavior. Note: the id scheme changed (12-char grid fingerprint vs 8-char cell hash); no committed registries existed at fix time, so there is no migration.

Location: [src/policy_registry.py:146](/Users/inigo_ocariz_laptop/src/scoring_tools/src/policy_registry.py:146).

**Trigger.** Bin edges change while the accepted integer-cell coordinates remain the same.

**Verified behavior.** Changing a cutpoint from 50 to 80 generates the same ID. Registering and promoting the new entry leaves one policy in the registry and retains the old champion edge=50.

**Cause and consequence.** The ID hashes accepted coordinates without their grid definition. register_policy treats an existing ID as a no-op and can promote the stale entry.

**Suggested correction.** Fingerprint ordered axis names, raw score sources, cutpoints and accepted cells together. Give genuinely different score regions different policy IDs.

**Prior audit context.** The edge-comparison guard detects the stale champion later but re-registering does not repair it.

## F5 — P2: Model compatibility checks omit raw score sources

> **Status: FIXED (2026-09-08).** Model metadata now persists `bin_sources` (bin variable → raw source column), threaded from the run config through `inference_pipeline` into `_save_model_to_disk`, and `validate_reused_model_config` refuses a reused model whose recorded source differs from the current config's `source_col` (same warn-and-proceed posture as pre-pin models for legacy metadata without the field). The V1 policy registry's matching gap was closed in F4 (`PolicyEntry.bin_sources` participates in the grid fingerprint). The reproduction below asserts the refusal and the legacy posture.

Location: [src/persistence.py:400](/Users/inigo_ocariz_laptop/src/scoring_tools/src/persistence.py:400).

**Trigger.** A BinConfig changes source_col while keeping its output name and bin_edges.

**Verified behavior.** Both configurations pass validate_reused_model_config. Real binning maps the same application from bin 1 to bin 2 when its old_score=10 and new_score=90 at cutpoint 50.

**Cause and consequence.** Saved metadata and reuse validation record inference variable names and edge values, but not the raw source columns that give them meaning. The V1 policy registry also omits those source definitions.

**Suggested correction.** Persist and validate the full input-to-bin mapping for reused models and frozen policies; include it in compatibility fingerprints.

**Prior audit context.** Additional gap in the grid identity covered by TODO #40. V2's Axis contract already captures source.

## F6 — P2: Cell confidence intervals refit the hurdle model on aggregated targets

> **Status: FIXED (2026-09-08).** `compute_cell_level_ci`'s CV folds now use the same per-loan hurdle training branch as model selection and the final fit (fit on `_hurdle_r`/`_hurdle_w` per-loan rows when the template is a `HurdleRegressor`), so a hurdle winner gets real cell CIs instead of a silently-dropped result or a different aggregated-fit estimator. The reproduction below asserts the CI frame is produced.

Location: [src/inference_optimized.py:980](/Users/inigo_ocariz_laptop/src/scoring_tools/src/inference_optimized.py:980).

**Trigger.** The per-loan HurdleRegressor wins model selection.

**Verified behavior.** The per-loan fit succeeds on 600 loans containing zero and nonzero outcomes. compute_cell_level_ci then raises 'at least 2 classes' because every aggregated training cell has a positive rate.

**Cause and consequence.** The CI path unconditionally fits the estimator on bin-level means, unlike selection, final fitting and held-out evaluation. The outer nonblocking catch drops the CI result; when bin means include zeros, it instead evaluates a different model.

**Suggested correction.** Use the same per-loan hurdle training branch inside each CI fold, or share one estimator-fit routine across all paths.

**Prior audit context.** Residual of the TODO #6 per-loan migration. Conditional on model_hurdle_per_loan=true and a hurdle winner.

## Verification and limits

- Full repository suite: **1,700 passed, 2 skipped, 4 warnings**. Ruff: **all checks passed**.
- All six synthetic reproductions passed their defect assertions. Run `uv run python reports/validation/code_audit_2026_09_08_checks.py` from the repository root. These assertions document current bugs and should be inverted or replaced when fixes land.
- Reproductions use temporary directories and synthetic frames. F1 mocks the unused reject prediction branch while exercising real booked aggregation, MILP and M4 accounting. F3 mocks deserialization to isolate the real companion-file selection. The remaining cases exercise their real functions.
- Production impact is conditional on each trigger; no numerical impact on the current SAS portfolio was measured. Do not interpret the synthetic percentages as current portfolio estimates.
- No additional confirmed defect was found in the reviewed V2 paths. V2 remains a development workflow with incomplete real-data validation, as its documentation states; this review does not change that status.

## Existing limitations kept separate

TODO #52's MR maturity issues remain tracked and deferred: summary H3 aggregates are not consistently age-filtered, and the configured MR window end remains the maturity anchor. M5 reference re-pinning remains deferred under the dataset-change decision. This audit does not change either decision.

The documented production-weighted global-allocation convention, post-selection holdout optimism, conditional frontier bootstrap, and the H3-to-H6 ecological-transfer caveat were not relabeled as new discoveries. HRI numerators are documented as sparse by design, so their nulls were not automatically treated as missing defaults.

A smaller issue noticed during tracing: the winner holdout call still passes `DEFAULT_Z_THRESHOLD` at [src/inference_optimized.py:1676](/Users/inigo_ocariz_laptop/src/scoring_tools/src/inference_optimized.py:1676), whereas final fitting uses the configured threshold. For nondefault values, that report evaluates a different outlier-handling procedure. This is source-verified only and is not included in the six reproduced findings.

