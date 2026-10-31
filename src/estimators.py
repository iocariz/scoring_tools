"""Compatibility shim: models pickled before the package split reference ``src.estimators``.

Keep until every saved ``model.pkl`` has been retrained; new code imports ``src.model.estimators``.
"""

from src.model.estimators import *  # noqa: F403
from src.model.estimators import HurdleRegressor, TweedieGLM  # noqa: F401
