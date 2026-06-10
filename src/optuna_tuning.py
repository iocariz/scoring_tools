"""
Optuna hyperparameter tuning for tree-based monotonic models.
"""

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.base import clone as sklearn_clone
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBRegressor

from src.constants import DEFAULT_RANDOM_STATE
from src.estimators import HurdleRegressor, TweedieGLM, prepare_model_input

# Disable optuna's default trial-level logging to keep the console clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _get_regression_weights(processed_data: pd.DataFrame):
    for col in ("todu_amt_pile_h6", "oa_amt_h0", "n_observations"):
        if col in processed_data.columns:
            return processed_data[col]
    return None


def _stratified_cv_splitter(raw_data: pd.DataFrame, cv_folds: int, random_state: int, indicators: list[str]):
    """Create a StratifiedKFold splitter using a binarized risk indicator.

    Credit risk data is heavily imbalanced (5-15% bad rate).  Standard KFold
    produces folds with highly variable bad-rate distributions, biasing model
    selection.  Stratifying on a binary default indicator keeps the bad rate
    stable across folds.

    Falls back to plain KFold when the indicator column is missing or has only
    one class (e.g., synthetic test data).
    """
    strat_col = indicators[0] if indicators else None
    if strat_col and strat_col in raw_data.columns:
        y_strat = (raw_data[strat_col] > 0).astype(int)
        if y_strat.nunique() >= 2 and y_strat.sum() >= cv_folds and (len(y_strat) - y_strat.sum()) >= cv_folds:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            return skf.split(raw_data, y_strat)
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return kfold.split(raw_data)


def tune_tree_models(
    raw_data: pd.DataFrame,
    bins: tuple,
    variables: list[str],
    indicators: list[str],
    target_var: str,
    multiplier: float,
    z_threshold: float,
    cv_folds: int = 5,
    n_trials: int = 30,
    random_state: int = DEFAULT_RANDOM_STATE,
    directions: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run Optuna to find best hyperparameters for XGBoost and LightGBM models.
    Both models will be constrained to exhibit monotonic behavior over variables.

    Returns:
        results_df: DataFrame containing CV results for the best models
        best_models: Dictionary of optimized (un-fitted) models
    """
    logger.info(f"Starting Optuna hyperparameter tuning for XGBoost and LightGBM ({n_trials} trials each)...")

    # Define monotonic constraints from directions (default -1 = decreasing risk)
    n_features = len(variables)
    if directions:
        xgb_monotone = tuple(directions.get(v, -1) for v in variables)
        lgb_monotone = [directions.get(v, -1) for v in variables]
    else:
        xgb_monotone = tuple([-1] * n_features)
        lgb_monotone = list([-1] * n_features)

    # Hyperparameters are tuned by k-fold CV on the full data (tuning seed); the chosen params are
    # then scored on a FRESH-seed k-fold pass below (a different fold partition) — a selection metric
    # with a real standard error and no single reused holdout (audit #7). KNOWN RESIDUAL BIAS (#32c):
    # re-partitioning removes partition-overfit but not data-overfit — a best-of-N-trials config keeps
    # some winner's-curse optimism vs untuned models (nested CV would remove it at ~k× the cost; the
    # 1SE rule with the NB-corrected SE partially offsets it). The winner-only 20% report in
    # inference_optimized is post-selection (see evaluate_holdout_rmse).
    tuning_data = raw_data
    fresh_seed = random_state + 1000

    from src.inference_optimized import _nadeau_bengio_cv_se, compute_outlier_stats, process_dataset

    def eval_model(model, raw_eval, cv_folds, random_state):
        splits = _stratified_cv_splitter(raw_eval, cv_folds, random_state, indicators)
        scores = []
        for train_idx, val_idx in splits:
            raw_train, raw_val = raw_eval.iloc[train_idx].copy(), raw_eval.iloc[val_idx].copy()

            # Aggregate train/val completely independently to prevent validation leakage.
            # Outlier stats come from the TRAIN fold (audit #32a): filtering the val fold
            # by its own target dropped exactly the riskiest val bins from scoring.
            train_agg = process_dataset(
                raw_train, bins, variables, indicators, target_var, multiplier, variables, z_threshold
            )
            train_outlier_stats = compute_outlier_stats(train_agg, target_var) if len(train_agg) > 2 else None
            val_agg = process_dataset(
                raw_val,
                bins,
                variables,
                indicators,
                target_var,
                multiplier,
                variables,
                z_threshold,
                outlier_stats=train_outlier_stats,
            )

            # Skip fold if validation has too few bins for reliable R²
            if len(val_agg) < 3:
                continue

            # Extract features/targets
            X_train, y_train = train_agg[variables], train_agg[target_var]
            X_val, y_val = val_agg[variables], val_agg[target_var]

            w_train = _get_regression_weights(train_agg)
            w_val = _get_regression_weights(val_agg)

            # Clone to avoid leaking fitted state across folds
            model_clone = sklearn_clone(model)
            model_clone.fit(X_train, y_train, sample_weight=w_train)
            pred = model_clone.predict(X_val)
            if len(y_val) >= 2:
                scores.append(np.sqrt(mean_squared_error(y_val, pred, sample_weight=w_val)))
        if not scores:
            return np.inf, 0.0
        # Nadeau–Bengio corrected SE (audit #32d) — fold scores are correlated.
        return np.mean(scores), _nadeau_bengio_cv_se(np.std(scores, ddof=1), len(scores))

    # --- XGBoost Study ---
    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
            "monotone_constraints": xgb_monotone,
            "random_state": random_state,
            "n_jobs": -1,
        }
        model = XGBRegressor(**params)
        mean_score, _ = eval_model(model, tuning_data, cv_folds, random_state)
        return mean_score

    study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study_xgb.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=True)

    best_xgb_params = study_xgb.best_params
    best_xgb_params["monotone_constraints"] = xgb_monotone
    best_xgb_params["random_state"] = random_state
    best_xgb_params["n_jobs"] = -1

    xgb_model = XGBRegressor(**best_xgb_params)
    # Fresh-seed k-fold CV for the selection metric (real mean + SE; different folds than tuning).
    xgb_mean, xgb_std = eval_model(xgb_model, raw_data, cv_folds, fresh_seed)
    logger.info(
        f"Best XGBoost: inner CV RMSE={study_xgb.best_value:.4f}, fresh-seed CV RMSE={xgb_mean:.4f} ± {xgb_std:.4f}"
    )

    # --- LightGBM Study ---
    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 1, 3),
            "num_leaves": trial.suggest_int("num_leaves", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "monotone_constraints": lgb_monotone,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        model = LGBMRegressor(**params)
        mean_score, _ = eval_model(model, tuning_data, cv_folds, random_state)
        return mean_score

    study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study_lgb.optimize(objective_lgb, n_trials=n_trials, show_progress_bar=True)

    best_lgb_params = study_lgb.best_params
    best_lgb_params["monotone_constraints"] = lgb_monotone
    best_lgb_params["random_state"] = random_state
    best_lgb_params["verbose"] = -1
    best_lgb_params["n_jobs"] = -1

    lgb_model = LGBMRegressor(**best_lgb_params)
    lgb_mean, lgb_std = eval_model(lgb_model, raw_data, cv_folds, fresh_seed)
    logger.info(
        f"Best LightGBM: inner CV RMSE={study_lgb.best_value:.4f}, fresh-seed CV RMSE={lgb_mean:.4f} ± {lgb_std:.4f}"
    )

    models = {"XGBoost (Optuna Tuned)": xgb_model, "LightGBM (Optuna Tuned)": lgb_model}

    results = [
        {
            "Model": "XGBoost (Optuna Tuned)",
            "CV Mean RMSE": xgb_mean,
            "CV Std RMSE": xgb_std,
            "model_template": xgb_model,
        },
        {
            "Model": "LightGBM (Optuna Tuned)",
            "CV Mean RMSE": lgb_mean,
            "CV Std RMSE": lgb_std,
            "model_template": lgb_model,
        },
    ]
    results_df = pd.DataFrame(results)

    return results_df, models


def tune_linear_models(
    raw_data: pd.DataFrame,
    bins: tuple,
    variables: list[str],
    indicators: list[str],
    target_var: str,
    multiplier: float,
    z_threshold: float,
    var_reg: list[str],
    cv_folds: int = 5,
    n_trials: int = 20,
    include_hurdle: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, dict]:
    """
    Run Optuna to find best hyperparameters for standard linear and GLM models.

    Returns:
        results_df: DataFrame containing CV results
        best_models: Dictionary of optimized (un-fitted) models
    """
    logger.info(f"Starting Optuna hyperparameter tuning for Linear/GLM models ({n_trials} trials each)...")

    fresh_seed = random_state + 1000

    # Tuning runs k-fold CV on the full data (tuning seed); the chosen params are scored on a
    # FRESH-seed k-fold pass (different fold partition) for the selection metric — real SE, no single
    # reused holdout (audit #7). The winning model's genuinely held-out 20% report is computed once in
    # inference_optimized and is never used for selection.
    tuning_data = raw_data

    from src.inference_optimized import _nadeau_bengio_cv_se, compute_outlier_stats, process_dataset
    from src.models import transform_variables

    def _fit_hurdle_per_loan(model_clone, raw_train):
        """Fit a HurdleRegressor on PER-LOAN rows (audit #6): the per-loan ratio `_hurdle_r` has
        real zero mass (non-defaulting loans → 0) so the classifier is meaningful, and `_hurdle_w`
        (exposure) weights the severity stage so the cell prediction reconciles to the
        exposure-weighted aggregate b2. Predictions are still made/scored on aggregated bins."""
        raw_train_t = transform_variables(raw_train, variables)
        x_pl = prepare_model_input(raw_train_t, var_reg, model_clone)
        y_pl = raw_train["_hurdle_r"].to_numpy()
        w_pl = np.asarray(raw_train["_hurdle_w"], dtype=float)
        model_clone.fit(x_pl, y_pl, sample_weight=w_pl)

    def eval_model(model, raw_eval, cv_folds, random_state):
        splits = _stratified_cv_splitter(raw_eval, cv_folds, random_state, indicators)
        scores = []
        for train_idx, val_idx in splits:
            raw_train, raw_val = raw_eval.iloc[train_idx].copy(), raw_eval.iloc[val_idx].copy()

            # Aggregate train/val completely independently to prevent validation leakage.
            # Outlier stats come from the TRAIN fold (audit #32a): filtering the val fold
            # by its own target dropped exactly the riskiest val bins from scoring.
            train_agg = process_dataset(
                raw_train, bins, variables, indicators, target_var, multiplier, var_reg, z_threshold
            )
            train_outlier_stats = compute_outlier_stats(train_agg, target_var) if len(train_agg) > 2 else None
            val_agg = process_dataset(
                raw_val,
                bins,
                variables,
                indicators,
                target_var,
                multiplier,
                var_reg,
                z_threshold,
                outlier_stats=train_outlier_stats,
            )

            # Skip fold if validation has too few bins for reliable R²
            if len(val_agg) < 3:
                continue

            # Extract generated features (prepare_model_input appends exposure for TweedieGLM)
            X_train = prepare_model_input(train_agg, var_reg, model)
            X_val = prepare_model_input(val_agg, var_reg, model)
            y_train, y_val = train_agg[target_var], val_agg[target_var]

            w_train = _get_regression_weights(train_agg)
            w_val = _get_regression_weights(val_agg)

            try:
                model_clone = sklearn_clone(model)
                if isinstance(model, HurdleRegressor) and "_hurdle_r" in raw_train.columns:
                    _fit_hurdle_per_loan(model_clone, raw_train)
                else:
                    model_clone.fit(X_train, y_train, sample_weight=w_train)
                pred = model_clone.predict(X_val)
                if len(y_val) >= 2:
                    scores.append(np.sqrt(mean_squared_error(y_val, pred, sample_weight=w_val)))
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
                raise optuna.exceptions.TrialPruned() from e

        if not scores:
            raise optuna.exceptions.TrialPruned()

        # Nadeau–Bengio corrected SE (audit #32d) — fold scores are correlated.
        return np.mean(scores), _nadeau_bengio_cv_se(np.std(scores, ddof=1), len(scores))

    def safe_eval(model):
        try:
            return eval_model(model, tuning_data, cv_folds, random_state)
        except Exception:
            logger.debug(f"Model evaluation failed for {type(model).__name__}: {model}", exc_info=True)
            return np.inf, 0.0

    def safe_eval_fresh(model):
        # Selection metric: fresh-seed k-fold CV (mean + real SE), a different fold partition than the
        # tuning objective so it isn't scored on the exact folds it was tuned on (audit #7).
        try:
            return eval_model(model, raw_data, cv_folds, fresh_seed)
        except Exception:
            logger.debug(f"Fresh-seed evaluation failed for {type(model).__name__}: {model}", exc_info=True)
            return np.inf, 0.0

    results = []
    models = {}

    def optimize_and_evaluate(objective_func, create_model_func, name_func):
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
        try:
            study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)
            if study.best_value == np.inf:
                return
            best_model = create_model_func(study.best_params)
            mean_rmse, std_rmse = safe_eval_fresh(best_model)
            if mean_rmse < np.inf:
                name = name_func(study.best_params)
                results.append(
                    {"Model": name, "CV Mean RMSE": mean_rmse, "CV Std RMSE": std_rmse, "model_template": best_model}
                )
                models[name] = best_model
        except ValueError:
            # Handle cases where all trials were pruned and best_params doesn't exist
            pass

    # 1. LinearRegression (baseline, no tuning needed)
    lin_model = LinearRegression(fit_intercept=True)
    lin_mean, lin_std = safe_eval_fresh(lin_model)
    if lin_mean < np.inf:
        results.append(
            {
                "Model": "Linear Regression",
                "CV Mean RMSE": lin_mean,
                "CV Std RMSE": lin_std,
                "model_template": lin_model,
            }
        )
        models["Linear Regression"] = lin_model

    # 2. Ridge
    def objective_ridge(trial):
        alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=random_state)
        mean_score, _ = safe_eval(model)
        return mean_score

    optimize_and_evaluate(
        objective_func=objective_ridge,
        create_model_func=lambda p: Ridge(alpha=p["alpha"], fit_intercept=True, random_state=random_state),
        name_func=lambda p: f"Ridge (Optuna Tuned α={p['alpha']:.3f})",
    )

    # 3. Lasso
    def objective_lasso(trial):
        alpha = trial.suggest_float("alpha", 0.001, 1.0, log=True)
        model = Lasso(alpha=alpha, fit_intercept=True, random_state=random_state)
        mean_score, _ = safe_eval(model)
        return mean_score

    optimize_and_evaluate(
        objective_func=objective_lasso,
        create_model_func=lambda p: Lasso(alpha=p["alpha"], fit_intercept=True, random_state=random_state),
        name_func=lambda p: f"Lasso (Optuna Tuned α={p['alpha']:.3f})",
    )

    # 4. ElasticNet
    def objective_enet(trial):
        alpha = trial.suggest_float("alpha", 0.001, 1.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.01, 0.99)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, random_state=random_state)
        mean_score, _ = safe_eval(model)
        return mean_score

    optimize_and_evaluate(
        objective_func=objective_enet,
        create_model_func=lambda p: ElasticNet(
            alpha=p["alpha"], l1_ratio=p["l1_ratio"], fit_intercept=True, random_state=random_state
        ),
        name_func=lambda p: f"ElasticNet (Optuna Tuned α={p['alpha']:.3f}, l1={p['l1_ratio']:.2f})",
    )

    # 5. TweedieGLM — exposure-weighted rate model (exposure enters as sample_weight, not a
    #    penalized log-exposure feature; audit #8). The weight is supplied by eval_model via
    #    _get_regression_weights (todu_amt_pile_h6).
    def objective_tweedie(trial):
        power = trial.suggest_float("power", 1.01, 1.99)
        alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
        model = TweedieGLM(power=power, alpha=alpha, link="log")
        mean_score, _ = safe_eval(model)
        return mean_score

    optimize_and_evaluate(
        objective_func=objective_tweedie,
        create_model_func=lambda p: TweedieGLM(power=p["power"], alpha=p["alpha"], link="log"),
        name_func=lambda p: f"Tweedie (Optuna Tuned p={p['power']:.2f}, α={p['alpha']:.2f})",
    )

    if include_hurdle:
        # 6. Hurdle-Ridge
        def objective_hurdle_ridge(trial):
            alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
            model = HurdleRegressor(
                classifier=LogisticRegression(max_iter=1000, random_state=random_state),
                regressor=Ridge(alpha=alpha, fit_intercept=True, random_state=random_state),
            )
            mean_score, _ = safe_eval(model)
            return mean_score

        optimize_and_evaluate(
            objective_func=objective_hurdle_ridge,
            create_model_func=lambda p: HurdleRegressor(
                classifier=LogisticRegression(max_iter=1000, random_state=random_state),
                regressor=Ridge(alpha=p["alpha"], fit_intercept=True, random_state=random_state),
            ),
            name_func=lambda p: f"Hurdle-Ridge (Optuna Tuned α={p['alpha']:.3f})",
        )

        # 7. Hurdle-Lasso
        def objective_hurdle_lasso(trial):
            alpha = trial.suggest_float("alpha", 0.001, 1.0, log=True)
            model = HurdleRegressor(
                classifier=LogisticRegression(max_iter=1000, random_state=random_state),
                regressor=Lasso(alpha=alpha, fit_intercept=True, random_state=random_state),
            )
            mean_score, _ = safe_eval(model)
            return mean_score

        optimize_and_evaluate(
            objective_func=objective_hurdle_lasso,
            create_model_func=lambda p: HurdleRegressor(
                classifier=LogisticRegression(max_iter=1000, random_state=random_state),
                regressor=Lasso(alpha=p["alpha"], fit_intercept=True, random_state=random_state),
            ),
            name_func=lambda p: f"Hurdle-Lasso (Optuna Tuned α={p['alpha']:.3f})",
        )

    if not results:
        raise RuntimeError("All Optuna linear/GLM model tuned trials failed or were pruned.")

    results_df = pd.DataFrame(results).sort_values("CV Mean RMSE", ascending=True)
    # Ensure columns match expected upstream layout
    results_df = results_df[["Model", "CV Mean RMSE", "CV Std RMSE", "model_template"]]

    return results_df, models
