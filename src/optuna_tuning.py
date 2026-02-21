"""
Optuna hyperparameter tuning for tree-based monotonic models.
"""

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.constants import DEFAULT_RANDOM_STATE
from src.estimators import HurdleRegressor, TweedieGLM

# Disable optuna's default trial-level logging to keep the console clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


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

    fresh_seed = random_state + 1000

    from sklearn.base import clone as sklearn_clone

    from src.inference_optimized import process_dataset

    def eval_model(model, raw_eval, cv_folds, random_state):
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, val_idx in kfold.split(raw_eval):
            raw_train, raw_val = raw_eval.iloc[train_idx].copy(), raw_eval.iloc[val_idx].copy()

            # Aggregate train/val completely independently to prevent validation leakage
            train_agg = process_dataset(
                raw_train, bins, variables, indicators, target_var, multiplier, variables, z_threshold
            )
            val_agg = process_dataset(
                raw_val, bins, variables, indicators, target_var, multiplier, variables, z_threshold
            )

            # Skip fold if validation has too few bins for reliable R²
            if len(val_agg) < 3:
                continue

            # Extract features/targets
            X_train, y_train = train_agg[variables], train_agg[target_var]
            X_val, y_val = val_agg[variables], val_agg[target_var]

            w_train = train_agg["todu_amt_pile_h6"] if "todu_amt_pile_h6" in train_agg.columns else None
            w_val = val_agg["todu_amt_pile_h6"] if "todu_amt_pile_h6" in val_agg.columns else None

            # Clone to avoid leaking fitted state across folds
            model_clone = sklearn_clone(model)
            model_clone.fit(X_train, y_train, sample_weight=w_train)
            pred = model_clone.predict(X_val)
            if len(y_val) >= 2:
                scores.append(np.sqrt(mean_squared_error(y_val, pred, sample_weight=w_val)))
        if not scores:
            return np.inf, 0.0
        return np.mean(scores), np.std(scores, ddof=1) / np.sqrt(len(scores))

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
        mean_score, _ = eval_model(model, raw_data, cv_folds, random_state)
        return mean_score

    study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study_xgb.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=True)

    best_xgb_params = study_xgb.best_params
    best_xgb_params["monotone_constraints"] = xgb_monotone
    best_xgb_params["random_state"] = random_state
    best_xgb_params["n_jobs"] = -1

    xgb_model = XGBRegressor(**best_xgb_params)
    # Fresh-seed eval mitigates fold-level overfitting but is not fully nested CV;
    # R² may be slightly optimistic due to hyperparameter selection on the same data.
    xgb_mean, xgb_std = eval_model(xgb_model, raw_data, cv_folds, fresh_seed)
    logger.info(f"Best XGBoost CV RMSE: {xgb_mean:.4f} ± {xgb_std:.4f} (fresh-seed, not nested CV)")

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
        mean_score, _ = eval_model(model, raw_data, cv_folds, random_state)
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
    logger.info(f"Best LightGBM CV RMSE: {lgb_mean:.4f} ± {lgb_std:.4f} (fresh-seed, not nested CV)")

    models = {"XGBoost (Optuna Tuned)": xgb_model, "LightGBM (Optuna Tuned)": lgb_model}

    results = [
        {"Model": "XGBoost (Optuna Tuned)", "CV Mean RMSE": xgb_mean, "CV Std RMSE": xgb_std, "model_template": xgb_model},
        {"Model": "LightGBM (Optuna Tuned)", "CV Mean RMSE": lgb_mean, "CV Std RMSE": lgb_std, "model_template": lgb_model},
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
    include_hurdle: bool = True,
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

    from src.inference_optimized import process_dataset

    def eval_model(model, raw_eval, cv_folds, random_state):
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, val_idx in kfold.split(raw_eval):
            raw_train, raw_val = raw_eval.iloc[train_idx].copy(), raw_eval.iloc[val_idx].copy()

            # Aggregate train/val completely independently to prevent validation leakage
            train_agg = process_dataset(
                raw_train, bins, variables, indicators, target_var, multiplier, var_reg, z_threshold
            )
            val_agg = process_dataset(
                raw_val, bins, variables, indicators, target_var, multiplier, var_reg, z_threshold
            )

            # Skip fold if validation has too few bins for reliable R²
            if len(val_agg) < 3:
                continue

            # Extract generated features
            X_train, y_train = train_agg[var_reg], train_agg[target_var]
            X_val, y_val = val_agg[var_reg], val_agg[target_var]

            w_train = train_agg["todu_amt_pile_h6"] if "todu_amt_pile_h6" in train_agg.columns else None
            w_val = val_agg["todu_amt_pile_h6"] if "todu_amt_pile_h6" in val_agg.columns else None

            try:
                model.fit(X_train, y_train, sample_weight=w_train)
                pred = model.predict(X_val)
                if len(y_val) >= 2:
                    scores.append(np.sqrt(mean_squared_error(y_val, pred, sample_weight=w_val)))
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
                raise optuna.exceptions.TrialPruned() from e

        if not scores:
            raise optuna.exceptions.TrialPruned()

        return np.mean(scores), np.std(scores, ddof=1) / np.sqrt(len(scores))

    def safe_eval(model):
        try:
            return eval_model(model, raw_data, cv_folds, random_state)
        except Exception:
            return np.inf, 0.0

    def safe_eval_fresh(model):
        try:
            return eval_model(model, raw_data, cv_folds, fresh_seed)
        except Exception:
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
            {"Model": "Linear Regression", "CV Mean RMSE": lin_mean, "CV Std RMSE": lin_std, "model_template": lin_model}
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

    # 5. TweedieGLM
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
