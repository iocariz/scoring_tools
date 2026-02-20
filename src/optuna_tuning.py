"""
Optuna hyperparameter tuning for tree-based monotonic models.
"""
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge

from src.constants import DEFAULT_RANDOM_STATE
from src.estimators import HurdleRegressor, TweedieGLM

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_tree_models(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series | None,
    cv_folds: int = 5,
    n_trials: int = 30,
    random_state: int = DEFAULT_RANDOM_STATE
) -> tuple[pd.DataFrame, dict]:
    """
    Run Optuna to find best hyperparameters for XGBoost and LightGBM models.
    Both models will be constrained to exhibit monotonic behavior over variables.
    
    Returns:
        results_df: DataFrame containing CV results for the best models
        best_models: Dictionary of optimized (un-fitted) models
    """
    logger.info(f"Starting Optuna hyperparameter tuning for XGBoost and LightGBM ({n_trials} trials each)...")

    # Define generic monotonic constraints (-1 for all features)
    # Higher score -> lower risk => negative monotonicity
    n_features = X.shape[1]
    xgb_monotone = tuple([-1] * n_features)
    lgb_monotone = list([-1] * n_features)

    def eval_model(model, X_eval, y_eval, weights_eval, cv_folds, random_state):
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, val_idx in kfold.split(X_eval):
            X_train, X_val = X_eval.iloc[train_idx], X_eval.iloc[val_idx]
            y_train, y_val = y_eval.iloc[train_idx], y_eval.iloc[val_idx]
            w_train = weights_eval.iloc[train_idx] if weights_eval is not None else None
            w_val = weights_eval.iloc[val_idx] if weights_eval is not None else None

            model.fit(X_train, y_train, sample_weight=w_train)
            pred = model.predict(X_val)
            scores.append(r2_score(y_val, pred, sample_weight=w_val))
        return np.mean(scores), np.std(scores)

    # --- XGBoost Study ---
    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "monotone_constraints": xgb_monotone,
            "random_state": random_state,
            "n_jobs": -1
        }
        model = XGBRegressor(**params)
        mean_score, _ = eval_model(model, X, y, weights, cv_folds, random_state)
        return mean_score

    study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study_xgb.optimize(objective_xgb, n_trials=n_trials)

    best_xgb_params = study_xgb.best_params
    best_xgb_params["monotone_constraints"] = xgb_monotone
    best_xgb_params["random_state"] = random_state
    best_xgb_params["n_jobs"] = -1

    xgb_model = XGBRegressor(**best_xgb_params)
    xgb_mean, xgb_std = eval_model(xgb_model, X, y, weights, cv_folds, random_state)
    logger.info(f"Best XGBoost CV R²: {xgb_mean:.4f} ± {xgb_std:.4f}")

    # --- LightGBM Study ---
    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "monotone_constraints": lgb_monotone,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1
        }
        model = LGBMRegressor(**params)
        mean_score, _ = eval_model(model, X, y, weights, cv_folds, random_state)
        return mean_score

    study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study_lgb.optimize(objective_lgb, n_trials=n_trials)

    best_lgb_params = study_lgb.best_params
    best_lgb_params["monotone_constraints"] = lgb_monotone
    best_lgb_params["random_state"] = random_state
    best_lgb_params["verbose"] = -1
    best_lgb_params["n_jobs"] = -1

    lgb_model = LGBMRegressor(**best_lgb_params)
    lgb_mean, lgb_std = eval_model(lgb_model, X, y, weights, cv_folds, random_state)
    logger.info(f"Best LightGBM CV R²: {lgb_mean:.4f} ± {lgb_std:.4f}")

    models = {
        "XGBoost (Optuna Tuned)": xgb_model,
        "LightGBM (Optuna Tuned)": lgb_model
    }

    results = [
        {
            "Model": "XGBoost (Optuna Tuned)",
            "CV Mean R²": xgb_mean,
            "CV Std R²": xgb_std,
            "model_template": xgb_model
        },
        {
            "Model": "LightGBM (Optuna Tuned)",
            "CV Mean R²": lgb_mean,
            "CV Std R²": lgb_std,
            "model_template": lgb_model
        }
    ]
    results_df = pd.DataFrame(results)

    return results_df, models


def tune_linear_models(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series | None,
    cv_folds: int = 5,
    n_trials: int = 20,
    include_hurdle: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE
) -> tuple[pd.DataFrame, dict]:
    """
    Run Optuna to find best hyperparameters for standard linear and GLM models.
    
    Returns:
        results_df: DataFrame containing CV results
        best_models: Dictionary of optimized (un-fitted) models
    """
    logger.info(f"Starting Optuna hyperparameter tuning for Linear/GLM models ({n_trials} trials each)...")

    def eval_model(model, X_eval, y_eval, weights_eval, cv_folds, random_state):
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, val_idx in kfold.split(X_eval):
            X_train, X_val = X_eval.iloc[train_idx], X_eval.iloc[val_idx]
            y_train, y_val = y_eval.iloc[train_idx], y_eval.iloc[val_idx]
            w_train = weights_eval.iloc[train_idx] if weights_eval is not None else None
            w_val = weights_eval.iloc[val_idx] if weights_eval is not None else None

            try:
                model.fit(X_train, y_train, sample_weight=w_train)
                pred = model.predict(X_val)
                scores.append(r2_score(y_val, pred, sample_weight=w_val))
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                raise optuna.exceptions.TrialPruned()
        
        if not scores:
            raise optuna.exceptions.TrialPruned()
            
        return np.mean(scores), np.std(scores)

    def safe_eval(model):
        try:
            return eval_model(model, X, y, weights, cv_folds, random_state)
        except Exception:
            return -np.inf, 0.0

    results = []
    models = {}

    def optimize_and_evaluate(objective_func, create_model_func, name_func):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
        try:
            study.optimize(objective_func, n_trials=n_trials)
            if study.best_value == -np.inf:
                return
            best_model = create_model_func(study.best_params)
            mean_r2, std_r2 = safe_eval(best_model)
            if mean_r2 > -np.inf:
                name = name_func(study.best_params)
                results.append({
                    "Model": name, 
                    "CV Mean R²": mean_r2, 
                    "CV Std R²": std_r2, 
                    "model_template": best_model
                })
                models[name] = best_model
        except ValueError:
            # Handle cases where all trials were pruned and best_params doesn't exist
            pass

    # 1. LinearRegression (baseline, no tuning needed)
    lin_model = LinearRegression(fit_intercept=False)
    lin_mean, lin_std = safe_eval(lin_model)
    if lin_mean > -np.inf:
        results.append({"Model": "Linear Regression", "CV Mean R²": lin_mean, "CV Std R²": lin_std, "model_template": lin_model})
        models["Linear Regression"] = lin_model

    # 2. Ridge
    def objective_ridge(trial):
        alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
        model = Ridge(alpha=alpha, fit_intercept=False, random_state=random_state)
        mean_score, _ = eval_model(model, X, y, weights, cv_folds, random_state)
        return mean_score
    optimize_and_evaluate(
        objective_func=objective_ridge,
        create_model_func=lambda p: Ridge(alpha=p["alpha"], fit_intercept=False, random_state=random_state),
        name_func=lambda p: f"Ridge (Optuna Tuned α={p['alpha']:.3f})"
    )

    # 3. Lasso
    def objective_lasso(trial):
        alpha = trial.suggest_float("alpha", 0.001, 1.0, log=True)
        model = Lasso(alpha=alpha, fit_intercept=False, random_state=random_state)
        mean_score, _ = eval_model(model, X, y, weights, cv_folds, random_state)
        return mean_score
    optimize_and_evaluate(
        objective_func=objective_lasso,
        create_model_func=lambda p: Lasso(alpha=p["alpha"], fit_intercept=False, random_state=random_state),
        name_func=lambda p: f"Lasso (Optuna Tuned α={p['alpha']:.3f})"
    )

    # 4. ElasticNet
    def objective_enet(trial):
        alpha = trial.suggest_float("alpha", 0.001, 1.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.01, 0.99)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, random_state=random_state)
        mean_score, _ = eval_model(model, X, y, weights, cv_folds, random_state)
        return mean_score
    optimize_and_evaluate(
        objective_func=objective_enet,
        create_model_func=lambda p: ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], fit_intercept=False, random_state=random_state),
        name_func=lambda p: f"ElasticNet (Optuna Tuned α={p['alpha']:.3f}, l1={p['l1_ratio']:.2f})"
    )

    # 5. TweedieGLM
    def objective_tweedie(trial):
        power = trial.suggest_float("power", 1.01, 1.99)
        alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
        model = TweedieGLM(power=power, alpha=alpha, link="log")
        mean_score, _ = eval_model(model, X, y, weights, cv_folds, random_state)
        return mean_score
    optimize_and_evaluate(
        objective_func=objective_tweedie,
        create_model_func=lambda p: TweedieGLM(power=p["power"], alpha=p["alpha"], link="log"),
        name_func=lambda p: f"Tweedie (Optuna Tuned p={p['power']:.2f}, α={p['alpha']:.2f})"
    )

    if include_hurdle:
        # 6. Hurdle-Ridge
        def objective_hurdle_ridge(trial):
            alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
            model = HurdleRegressor(
                classifier=LogisticRegression(max_iter=1000, random_state=random_state),
                regressor=Ridge(alpha=alpha, fit_intercept=False, random_state=random_state)
            )
            mean_score, _ = eval_model(model, X, y, weights, cv_folds, random_state)
            return mean_score
        optimize_and_evaluate(
            objective_func=objective_hurdle_ridge,
            create_model_func=lambda p: HurdleRegressor(
                classifier=LogisticRegression(max_iter=1000, random_state=random_state),
                regressor=Ridge(alpha=p["alpha"], fit_intercept=False, random_state=random_state)
            ),
            name_func=lambda p: f"Hurdle-Ridge (Optuna Tuned α={p['alpha']:.3f})"
        )

        # 7. Hurdle-Lasso
        def objective_hurdle_lasso(trial):
            alpha = trial.suggest_float("alpha", 0.001, 1.0, log=True)
            model = HurdleRegressor(
                classifier=LogisticRegression(max_iter=1000, random_state=random_state),
                regressor=Lasso(alpha=alpha, fit_intercept=False, random_state=random_state)
            )
            mean_score, _ = eval_model(model, X, y, weights, cv_folds, random_state)
            return mean_score
        optimize_and_evaluate(
            objective_func=objective_hurdle_lasso,
            create_model_func=lambda p: HurdleRegressor(
                classifier=LogisticRegression(max_iter=1000, random_state=random_state),
                regressor=Lasso(alpha=p["alpha"], fit_intercept=False, random_state=random_state)
            ),
            name_func=lambda p: f"Hurdle-Lasso (Optuna Tuned α={p['alpha']:.3f})"
        )

    if not results:
        raise RuntimeError("All Optuna linear/GLM model tuned trials failed or were pruned.")
        
    results_df = pd.DataFrame(results).sort_values("CV Mean R²", ascending=False)
    # Ensure columns match expected upstream layout
    results_df = results_df[["Model", "CV Mean R²", "CV Std R²", "model_template"]]
    
    return results_df, models
