"""Thin re-exporter for the pipeline-optimization phase modules.

R2b-iv (todo #63) split the former 1300-line
``src/pipeline/optimization.py`` into four focused modules:

  - :mod:`src.pipeline.cutoff_optimization` — :class:`OptimizationResult`,
    :func:`run_optimization_phase`.
  - :mod:`src.pipeline.scenarios` — :func:`run_scenario_analysis`,
    :func:`build_scenario_list`, :func:`compute_mr_annual_coef`,
    :func:`save_cutoff_summaries`.
  - :mod:`src.pipeline.sensitivity` — :func:`run_sensitivity_phase`.
  - :mod:`src.pipeline.ri_optimizer` — :func:`run_ri_optimizer_phase`.

This module keeps the legacy import path available so that
``from src.pipeline.optimization import run_optimization_phase`` etc.
continue to work unchanged. New code should import from the focused
modules above.
"""

from src.pipeline.cutoff_optimization import OptimizationResult, run_optimization_phase
from src.pipeline.ri_optimizer import run_ri_optimizer_phase
from src.pipeline.scenarios import (
    build_scenario_list,
    compute_mr_annual_coef,
    run_scenario_analysis,
    save_cutoff_summaries,
)
from src.pipeline.sensitivity import run_sensitivity_phase

__all__ = [
    "OptimizationResult",
    "build_scenario_list",
    "compute_mr_annual_coef",
    "run_optimization_phase",
    "run_ri_optimizer_phase",
    "run_scenario_analysis",
    "run_sensitivity_phase",
    "save_cutoff_summaries",
]
