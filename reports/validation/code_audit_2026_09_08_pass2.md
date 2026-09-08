# Code audit — second pass, 2026-09-08

Five additional findings were confirmed by the review. **F8 and F9 (both P1) are now fixed in the working tree** at the user's request; the three P2 findings and the related policy-ID traceability defect remain open. The original review was read-only; the follow-up changed the relevant optimization and policy-comparison code.

Reviewed the working tree at `6f719b1f2656f229e500db0f0dfe22128188a9cb`, including the local hurdle-CI change and uncommitted V2 files. Context: `CLAUDE.md`, `README.md`, `todo-list.md`, repository guidance, the earlier audit and its updated reproductions. Findings retain F7–F11 identifiers to distinguish them from F1–F6 in the earlier report. This report takes precedence over the earlier report's claims that exposure-model persistence and registry source compatibility are fully fixed.

## F8 — P1: A cell with no usable booked outcomes contributes production without risk

> **FIXED, 2026-09-08.** `CellGrid.usable_risk_mask` separates outcome availability from record presence. MILP/GA reject unsupported cells; frontier fallbacks preserve the exclusions; must-accept conflicts fail and legacy enumeration cannot bypass the gate. Fixed cutoffs that accept an observed cell without usable risk fail explicitly. Source production and baseline accounting remain intact. The reproduction now accepts only €100 at 1% risk and rejects the unmeasured €10,000 cell.

Locations: [booked aggregation](/Users/inigo_ocariz_laptop/src/scoring_tools/src/inference_optimized.py:2127), [MILP risk coefficients and denominator guard](/Users/inigo_ocariz_laptop/src/scoring_tools/src/optimization_utils.py:509).

**Trigger.** A booked cell has positive production but every H6 numerator/denominator pair is missing. Another accepted cell has positive, complete exposure.

**Reproduction.** The real aggregation/merge pipeline and MILP accept both cells under a 2% cap. Reported production is €10,100 and reported risk is 1%, but €10,000 of that production comes from the cell with no usable outcomes. Its risk contributes nothing to the constraint. The full portfolio's risk is unknown; the synthetic 1% is supported only by the other cell.

**Cause.** Grouped sums turn all-missing outcomes into 0/0 while retaining production. The cell is considered observed because it has records. Both its linearized risk coefficients are zero, and the portfolio-level positive-denominator guard is satisfied by the other cell. Reject prediction fills rejected cells, leaving this booked cell unestimated. Joint masking from F1 fixes partial-pair dilution within an observed cell but does not address an entirely unobserved risk surface region.

**Correction.** Carry cell-level outcome availability separately from production. Refuse or constrain acceptance of cells without usable risk evidence, or supply an explicit, validated risk estimate with its provenance. Do not infer risk availability from record presence. Add coverage for a mixed grid containing both measured and entirely unmeasured cells.

## F9 — P1: Champion comparison ignores the champion's recorded score source

> **FIXED, 2026-09-08.** Comparison validates `bin_sources` on every optimization axis before binning or scoring. Changed, missing and partial legacy source mappings are refused; matching mappings retain the correct frozen-policy results. Registration and comparison share source extraction. The related P2 challenger-ID mismatch below remains open.

Location: [compare_segment compatibility checks and binning](/Users/inigo_ocariz_laptop/src/scoring_tools/run_policy_registry.py:203).

**Trigger.** Change `BinConfig.source_col` while preserving output variables, edges and integer cell coordinates. The champion was registered under the old source.

**Reproduction.** Register a champion using `old_score`, then compare under `new_score`. The real binning and comparison proceed without a compatibility warning. The champion is reported at 2% risk and €4,000 production; applying its frozen source mapping to the identical matured cohort gives 1% and €2,000.

**Cause.** `PolicyEntry.bin_sources` is now persisted and participates in registration fingerprints, but `compare_segment` checks only variables and edges. It applies both policies after transforming the data with the current source mapping. Recording source identity at registration does not enforce it at evaluation. This is the remaining registry-side gap from F4/F5; the model reuse guard itself now checks sources correctly.

**Correction.** Validate the full frozen mapping, including `bin_sources`, before evaluating a champion; handle legacy entries lacking that evidence explicitly. Alternatively, bin each policy under its own frozen definition and compare loan-level accepted sets. Do not silently interpret old coordinates using a different score.

**Related traceability defect (P2).** [The challenger ID](/Users/inigo_ocariz_laptop/src/scoring_tools/run_policy_registry.py:149) still uses the old 8-character accepted-cell hash. Registration now uses a 12-character grid fingerprint. The same current policy is called `audit_pass2-b42a73da` in the comparison and `audit_pass2-8309ca64581b` when registered. Comparison evidence therefore cannot be joined to the registry by its advertised challenger ID. Reuse the canonical identity builder.

## F11 — P2: Reject-inference pooling is order-dependent and is not the claimed isotonic projection

Location: [partial-order block merging](/Users/inigo_ocariz_laptop/src/scoring_tools/src/reject_inference.py:714).

**Trigger.** `reject_enforce_monotonicity=true` on a sparse multidimensional grid with branching order constraints. This toggle is enabled in the checked-in `config.toml`.

**Reproduction through `apply_parceling_adjustment`.** Use equal-evidence cells `(0,0)`, `(1,2)`, `(2,1)`, with initial multipliers `[2.98, 2.50, 1.02]`. The first cell must have a multiplier no larger than either other cell; the latter two are incomparable. The function returns `[2.1667, 2.1667, 2.1667]`, with unweighted squared error 2.087467. The feasible alternative `[2.00, 2.50, 2.00]` has strictly smaller squared error, 1.920800. Evidence weights are identical, so the weighted objective has the same ordering. Reordering the same input rows makes the function return that better alternative. The changed multipliers actually scale the default amounts sent downstream.

**Cause.** Greedily merging any violating pair of blocks cannot undo an unnecessary early merge. It guarantees eventual feasibility but does not establish the claimed least-squares optimum on a general partial order. Here every axis slice is a singleton, so the preceding axis-wise regression cannot explain away the counterexample. TODO #17's monotonicity guarantee holds in this example; its claim of a correct isotonic projection does not. Isotonic regression requires both respecting the order and minimizing the specified discrepancy, as defined by [Kyng, Rao and Sachdeva](https://arxiv.org/abs/1507.00710).

**Correction.** Use a solver for the weighted least-squares problem over the full partial order, using the original multipliers as the objective data. Test branching orders, row permutations and agreement with a small reference optimization problem. A chain-only regression test cannot establish correctness on a partial order. Revalidate any changed risk multipliers and resulting cutoffs.

## F7 — P2: New training still saves the exposure model outside its versioned directory

Location: [paired exposure save](/Users/inigo_ocariz_laptop/src/scoring_tools/src/pipeline/inference.py:236).

**Trigger.** Train a new model and later select it after another training has saved a newer model directory.

**Reproduction.** Use the real return value of `save_model_with_metadata` and execute the training orchestration's exposure-save branch. The versioned directory contains no `todu_model.joblib`; the shared parent directory does. After a second save, reuse of the first newly saved model raises the exposure-pairing error.

**Cause.** `save_model_with_metadata` returns a directory, and `inference_pipeline` propagates it as `model_path`. The new pairing code treats it as a file and writes to `Path(saved_model_file).parent`. The loader expects the companion inside the directory. The F3 load guard is working, but the save branch never creates the new-format pair that the guard relies on. Existing pairing tests manually create companions rather than exercising the save/load contract.

**Correction.** Write the exposure model and its integrity sidecar under the returned version directory. Add a two-training roundtrip that reuses the first model and verifies its original exposure model. Preserve the current refusal of unverifiable older legacy pairs.

## F10 — P2: HRI sensitivity silently re-optimizes b2

Locations: [sensitivity orchestration](/Users/inigo_ocariz_laptop/src/scoring_tools/src/pipeline/sensitivity.py:91), [risk perturbation and re-solve](/Users/inigo_ocariz_laptop/src/scoring_tools/src/sensitivity.py:104).

**Trigger.** `risk_indicator="hri_h6"` and `run_sensitivity=true`. The latter is enabled in the checked-in configuration; HRI remains an opt-in target.

**Reproduction.** An HRI policy constrained at 1% accepts only the first of two cells. With the actual frozen mask saved, the sensitivity phase at **0% perturbation** reports one reject-to-accept flip, €200 production and `new_risk=0.1`. That 0.1% is b2. The resulting policy's HRI is 4.25%, above its 1% target; an unchanged HRI optimization still accepts only the original cell.

**Cause.** The baseline now correctly comes from the shipped mask, but subsequent perturbations, MILP coefficients, target, multiplier and reported risk remain hard-coded to b2/`optimum_risk`. The cell-detail and marginal-impact outputs have the same basis mismatch. These outputs cannot describe HRI cutoff robustness.

**Correction.** Thread the selected indicator, numerator, denominator, multiplier and target through all sensitivity calculations. Until implemented, skip or reject this configuration with an explicit explanation, as other unsupported HRI paths already do. Verify zero-perturbation identity on a unique optimum and show which risk measure the sensitivity output uses.

## Verification, earlier findings and limits

- Full suite after the P1 fixes: **1,740 passed, 2 skipped, 4 warnings**, including **22 added regression cases**. `uv run ruff check .`: **passed**. The initial review had 1,718 passing tests.
- Run `uv run python reports/validation/code_audit_2026_09_08_pass2_checks.py`: F8/F9 now assert their corrected behavior; F7/F10/F11 still assert the open defects. The F9 reproduction also retains the related challenger-ID mismatch assertion.
- The updated F1–F6 reproduction script passes. F1's joint masking, F2's positive-exposure observability guard, F4's registration fingerprint, F5's model-source guard and the local F6 hurdle-CI fix work for their reproduced triggers. F3's loader behavior is fixed, with the save-side omission documented in F7. F9 covers the remaining comparison-stage source mismatch.
- F7 bypasses model fitting and deserialization, while exercising the actual persistence return value, exposure-file writes and reuse guard. F8 bypasses only prediction on an empty rejected-loan frame; aggregation, merging and MILP are real. F9 bypasses unrelated segment preprocessing while using real source binning, compatibility checks, matured holdout selection and risk bootstraps. F10 and F11 exercise their real public phase/functions without mocked calculations.
- The broader review covered preprocessing, model fitting/uncertainty, optimization, reject inference, HRI integration, model reuse and policy governance, with the previous review of the local V2 implementation carried forward. This is not a proof that the remaining code is bug-free.
- No fresh real-data impact measurement or model retraining was performed. Synthetic euros and rates demonstrate failure modes; they are not estimates for the current portfolio.
- Existing deferred MR maturity issue #52 and M5 reference re-pinning remain tracked separately. The documented production-weighted allocation convention and HRI sparse-by-design numerator semantics were respected. No additional V2 defect was confirmed; its development/validation status is unchanged.

The earlier report and its user-edited fix annotations were preserved. The follow-up fixes add regression coverage for aggregation, optimizer/fallback exclusions, fixed-policy conflicts, source-map compatibility and legacy refusal.

**Existing-output impact check (follow-up).** Inspected all seven current segment `data_summary_desagregado.csv` files and their accepted-cell sets without overwriting pipeline artifacts. All six optimizing segments have usable target-risk evidence in every observed cell, so the new evidence bounds leave their current optimization surfaces unchanged. The remaining segment (`direct-conso-known-premium`) runs in baseline mode: its 88 unsupported cells carry €0 production and its baseline behavior is preserved. This is a check of existing outputs, not a fresh SAS/model reproduction.
