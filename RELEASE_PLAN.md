# Release Plan

Sequenced delivery plan for the audit backlog (`todo_list.md`, items 1–87) and feature roadmap (`new_features.md`, items 1–28).

## Guiding principles

- **Security blockers ship alone** — no feature work in the same release as a CVE fix.
- **Hardening before features** — type safety, CI gates, and lockfile land before new methodology.
- **Decomposition before new code** — no new features land in monolithic functions; split first.
- **Foundations before consumers** — registry and DQ gate before drift detection and reporting.
- **One methodology release per train** — R3 and R5 don't overlap; reviewers need time.

Cross-references: items prefixed `#NN` are from `todo_list.md`; items prefixed `Feature #NN` are from `new_features.md`.

---

## R0 — Security patch (1–2 weeks, ships alone)

Cannot share a release with anything else. Small diff, hard gate.

**Scope**
- #43 Jinja2 `autoescape=False` (stored XSS in HTML reports)
- #44 `joblib.load` RCE via `--model-path`
- #45 9× `except Exception: pass` silent corruption
- #47 `debug=True` passthrough via `--debug` CLI
- #48 `gradio --share` public tunnel
- #49 PII rows logged at DEBUG

**Exit criteria**
- Red-team replay of each CVE fails.
- `security-reviewer` agent clean on the diff.
- No new files, no new features.

---

## R1 — Hardening foundation (2–3 weeks)

Unblocks confident iteration. No new features.

**Scope**
- Security: #46 `/static` route validation, #50 dashboard auth layer, #51 commit `uv.lock` + `uv lock --check` in CI
- Type safety batch: #52–#56 (Literal, base_path, bool|None, union-attr, PEP 484)
- Coverage floor: #59 set `--cov-fail-under=60` in CI (ratchets later)
- Hygiene: #60 git-artefact purge (40 MB), #61 print→logger, #62 main() docstring sync, #70 undocumented config fields, #75 dead code

**Exit criteria**
- `mypy src/` clean.
- `uv lock --check` green in CI.
- Fresh clone < 10 MB.
- CI fails below 60% coverage.

---

## R2 — Architecture decomposition (4–6 weeks)

Prerequisite for every downstream feature landing cleanly.

**Scope**
- #57 split `export_consolidated_excel` (2004 lines) and `process_mr_period` (943 lines) into per-responsibility helpers
- #63 split `src/pipeline/optimization.py` into `cutoff_optimization.py` / `scenarios.py` / `sensitivity.py`; introduce `OptimizationResult` dataclass
- #64 extract `src/dashboard_data.py`; retire `gradio_dashboard.py`
- #65 `CutoffSpec` value type; remove hardcoded 2-var branches
- #66 deprecation warnings on legacy `octroi_bins` / `efx_bins`
- #67 replace `os.chdir` with explicit `OutputPaths(base_dir=...)`
- Coverage lift (#58): `config_loader.py`, `pipeline/optimization.py`, `inference_optimized.py` → ≥70%; ratchet CI gate to 70%

**Exit criteria**
- No function > 300 lines in `src/consolidation.py` or `src/mr_pipeline.py`.
- Pipeline tests mock the MILP solver; run in < 60 s.
- CI fails below 70% coverage.

---

## R3 — Methodology quality pass (4–5 weeks)

Close statistical gaps affecting optimization correctness. Runs in parallel with R2 where possible (independent files).

**Scope — low-effort, high-clarity**
- #14 document/switch to BCa for Gini/KS CIs
- #17 config cross-validation (`@model_validator(mode="after")`)
- #21 TweedieGLM offset (document as approximate)
- #22 orthogonal polynomials on bin indices
- #23 one-SE rule compounding — nested CV or single joint selection
- #25, #80, #81 reject inference: exogeneity doc, EB prior, missing-bin handling
- #83–#87 silent data loss fixes (quantile ties, NaN in `pd.cut`, category collisions, audit validation, TweedieGLM predict fallback)

**Scope — larger items**
- #24 include MR H3 in H3→H6 curvature fit
- #34 out-of-time calibration (flag-gated)
- #42 run-quality gates (extrapolation share, fallback share)

**Exit criteria**
- Per-scenario HTML report banner surfaces extrapolation / fallback / DQ warnings.
- Cross-segment regression: point estimates move by < 0.2 pp on reference runs unless intentionally changed.

---

## R4 — Governance bedrock (4–5 weeks)

Other features depend on these.

**Scope**
- Feature #13 Data-Quality gate (blocks optimization on DQ failure)
- Feature #16 Versioned cutoff registry (4-hash tuple → deployment date)
- Feature #15 Fairness / disparate-impact analysis

**Exit criteria**
- Every deployed cutoff reproducible from `(config, model, data, cutoff)` hash.
- DQ gate blocks a known-bad run in CI regression.
- `fairness_report.html` section produced for every segment; 80% rule flagged when violated.

---

## R5 — Optimizer quality upgrade (6–8 weeks)

Highest-leverage block: changes which cutoff is actually optimal.

**Scope**
- Feature #1 Score calibration report (Hosmer-Lemeshow, decile calibration curve)
- Feature #14 Economic-value objective (closes #29)
- Feature #6 Robust optimization under uncertainty (closes #36; uses existing bootstrap infra)
- Feature #9 Out-of-time backtesting

**Exit criteria**
- At least one segment's recommended cutoff demonstrably shifts when P&L objective replaces volume — business-reviewed.
- 12-month backtest reported for every segment with cutoff-stability metric.
- Calibration report triggers model-refresh recommendation when miscalibration exceeds threshold.

---

## R6 — Reporting refresh (5–6 weeks)

After registry + KPI foundations exist.

**Scope — foundation first**
- Feature #26 single KPI module (`src/report_kpis.py`) — **must land before other R6 items**

**Scope — venues**
- Excel: Features #17 (executive tab), #18 (policy-diff), #19 (metadata-driven formatting)
- HTML: Features #20 (TL;DR), #21 (inline provenance), #22 (since-last-run diff), #23 (policy-spec JSON)
- Dashboard: Features #24 (deployed-vs-optimal overlay), #25 (cell drill-down)
- Cross-cutting: Features #27 (config-driven sections), #28 (comparative mode)

**Exit criteria**
- Excel, HTML, and dashboard agree on optimum risk to 4 dp on all reference runs.
- Dashboards consolidated to a single Dash app.
- Executive tab approved by a committee reviewer.

---

## R7+ — Research backlog (rolling, each a multi-sprint investigation)

Larger bets, not sequenced into fixed releases.

- Feature #4 bivariate parceling
- Feature #3 vintage-aware risk surface
- Feature #10 cutoff drift detection (depends on #16)
- Feature #11 swap impact tracking (depends on #16)
- Feature #2 champion-challenger (depends on Feature #28)
- Feature #5 multi-objective Pareto (NSGA-III)
- Feature #7 dynamic / rolling cutoff optimization
- Feature #8 automated segment discovery
- Feature #12 what-if scenario API

---

## Sizing assumptions

- One FTE-equivalent. Halve durations with two engineers on independent tracks.
- R2 and R3 parallelize cleanly (different files).
- R0 and R1 do not parallelize — they're the foundation everything else stacks on.
- Methodology items (R3, R5) need a second reviewer; budget ~20% review time.

## Explicit non-goals per release

- R0–R1: zero new features.
- R2: zero methodology changes.
- R5: no reporting work.
- R6: no optimizer changes.

## Release dependencies graph

```
R0 ─► R1 ─► R2 ─┬─► R4 ──► R5 ──► R6 ──► R7+
                │
                └─► R3 ──► R5
```

R3 can begin in parallel with R2 but must complete before R5 starts.
R4 must complete before R5's Feature #9 (backtesting needs the registry from Feature #16).
R6's Feature #28 (comparative mode) unblocks R7's Feature #2 (champion-challenger).

## Progress tracking

Each release closes a defined subset of `todo_list.md` or `new_features.md`. A release is Done when:

1. All scope items merged and deployed.
2. Exit criteria met and verified.
3. `todo_list.md` or `new_features.md` updated — items moved to the archive with a brief closure note.
4. Follow-up items (new audit findings surfaced during the release) added to the backlog, not silently absorbed.
