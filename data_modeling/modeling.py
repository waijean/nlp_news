from typing import List, Dict

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline

from utils.constants import CLASSIFIER
from utils.pipeline_abc import logger


def load_and_split_data(read_path: str, X_col: List[str], y_col: str):
    logger.info(f"Loading data from {read_path}")
    X = pd.read_parquet(read_path, columns=X_col)
    y = pd.read_parquet(read_path, columns=[y_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return X_train, X_test, y_train, y_test


def evaluate_cv_pipeline(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame, scoring: Dict
):
    logger.info("Evaluating pipeline")
    cross_validation = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        scoring=scoring,
        cv=cross_validation,
        n_jobs=-1,
        return_train_score=True,
        return_estimator=True,
    )
    # extract the classifier of first pipeline from cv_results
    fitted_classifier = cv_results["estimator"][0][CLASSIFIER]

    return fitted_classifier, cv_results


def evaluate_grid_search_pipeline(
    pipeline: Pipeline,
    param_grid,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    scoring: Dict,
):
    cross_validation = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cross_validation,
        refit="accuracy",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    gs = gs.fit(X_train, y_train)
    best = {
        "params": gs.best_params_,
        "metrics": gs.best_score_,
        "fitted_classifier": gs.best_estimator_,
    }
    return best, gs.cv_results_
