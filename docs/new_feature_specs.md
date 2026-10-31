# Feature Spec — Hierarchical Reporting Supersegments (Nested Subtotals)

**Status:** specified, not scheduled. Deferred on Inigo's call (2026-08-12) until the open audit errors are fixed.
**Context:** the new dataset/product setup introduces a channel hierarchy (e.g. `Direct→New→Fintonic`, `Direct→Known→A-B`, `Auto→OEM→Known→A-B`, ragged depths). Leaves map to segments today; the consolidated report can show only ONE level of subtotals (flat reporting supersegments), so channel subtotals and sub-channel subtotals cannot coexist in one report.

## Current state (verified 2026-08-12)

- Membership is resolved **per segment**: `resolve_reporting_supersegment(seg_config)` (`src/utils.py`) returns exactly one group name; `consolidation.py:1069-1074` builds `segment_to_supersegment` from it. Overlapping/nested groups are not expressible in config.
- The aggregation math is already composable: `aggregate_metrics(member_metrics_list)` (`consolidation.py:845`) pools risk numerators/denominators (`Σ todu_30ever_h6 / Σ todu_amt_pile_h6 × multiplier`) and production over any member list — a subtotal for any tree node is one call. The limitation is purely the grouping model.
- Aggregate rows are named `supersegment_<name>` (`consolidation.py:1108`); member rows `"{ss}/{seg}"`, standalone `"segment_<seg>"` (`:1150-1158`). Downstream consumers rely on these prefixes:
  - deck charts exclude `supersegment_*` + TOTAL (`presentation_charts.py`, audit fix #24 — prevents double-counting);
  - the audit production patch restricts to detail rows via `_detail_segment_mask` (`consolidation.py:757`, audit fix #25);
  - row ordering/labeling at `consolidation.py:1296,1314`.
- HTML portfolio grouping: `reporting.py` `_portfolio_group_list` (~`:804-816`) — note open finding **#45** (standalone segments dropped when any supersegment exists) lives here; fix or at least don't worsen it.

## Proposed design

### Config

`reporting_supersegment` accepts a **path** (slash-separated); a plain name remains a depth-1 path (full back-compat):

```toml
[segments.direct_known_ab]
segment_filter = "direct_known_ab"
reporting_supersegment = "direct/known"     # was: "direct"

[segments.auto_oem_known_ab]
reporting_supersegment = "auto/oem/known"
```

- `[reporting_supersegments.*]` blocks stay for metadata/back-compat but node membership is derived from the per-segment paths (today's `resolve_reporting_supersegment` precedence `reporting_supersegment > supersegment > None` unchanged).
- Ragged depths are fine: `direct_consolidation` can sit at `reporting_supersegment = "direct"` while siblings go deeper.
- Legacy `supersegment` field: treated as depth-1 for both modelling and reporting, unchanged.
- **Modelling supersegments are out of scope** — they stay flat (a model is trained on one pooled population; nesting has no meaning there).

### Aggregation

For every distinct **prefix** of every segment's path, emit one aggregate row over the segments whose path starts with that prefix:

- `supersegment_direct` = all `direct/**` members
- `supersegment_direct/known` = all `direct/known/**` members

Each computed with the existing `aggregate_metrics` (no math changes — pooled numerators/denominators, None-poisoning semantics preserved). TOTAL remains computed from the segment list directly (never by summing supersegment rows), so nested rows cannot double-count it.

### Naming & rendering

- Row `group` values keep the `supersegment_` prefix: `supersegment_direct/known`. This preserves the #24/#25 exclusion conventions for free — verify with regression tests that deck charts and the audit patch still exclude all nested rows.
- Member rows become `"{full_path}/{seg}"` (e.g. `direct/known/direct_known_ab`); `_detail_segment_mask` must match on the last path component.
- Excel/HTML: sort TOTAL → tree pre-order → leaves; indent labels by path depth (`consolidation.py:1296` label helper, `:1314` ordering lambda; `reporting.py` portfolio sections mirror it).
- Risk surfaces / score metrics / selection-bias supersegment groupings: **top-level node only** in v1 (document this); nested surfaces are a possible v2.

### Interactions & edge cases

- **Sort keys:** a node's row must sort before its children and after its parent; use the path tuple, not string sort (`direct/known` vs `direct2`).
- **A segment name must not collide with a node path**; validate at config load and fail loudly (extend the Pydantic/segments.toml validation).
- **Open finding #45** (`reporting.py`): standalone segments must appear alongside the tree, not be dropped — fixing #45 is a natural part of this PR.
- **Open finding #62** (`fillna(0)` pooled system-rejection rates): the nested aggregation must reuse `aggregate_metrics`' None-poisoning, not the patch-path re-derivation, or fix #62 first.
- **MR period + all scenarios** get the same nested rows (the loop at `consolidation.py:1077-1093` already iterates scenario × period).
- Consumers that read the consolidated CSV externally: nested rows are additive; anything that filtered `supersegment_` prefix keeps working. Announce the new `group` format in CLAUDE.md.

## Acceptance criteria

1. Existing flat configs produce a **byte-identical** consolidated CSV/Excel (back-compat golden test on current `output/` artifacts).
2. A path config produces one aggregate row per tree node; every node's risk equals `mult·Σnum/Σden` over its leaves and its production sums exactly; all nodes reconcile to TOTAL.
3. Deck charts and the audit patch exclude nested rows (regression tests extending the #24/#25 tests).
4. Ragged-depth tree renders correctly ordered/indented in Excel and HTML; standalone segments still appear (#45 fixed or covered).
5. Real-artifact before/after on the live `output/` demonstrating the nested rows.

## Non-goals (v1)

- Nested **modelling** supersegments.
- Nested risk-surface/score-metrics groupings (top-level only).
- Cross-file (multi-SAS) channel consolidation — separate problem; channels must be distinguishable in one file via `segment_cut_off` (see the fully-qualified leaf-code convention; keep leaf codes non-prefix-overlapping until audit #39 is fixed).
