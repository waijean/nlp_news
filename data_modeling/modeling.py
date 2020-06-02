from typing import List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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
        pipeline,
        X_train,
        y_train,
        scoring=scoring,
        cv=cross_validation,
        n_jobs=-1,
        return_train_score=True,
        return_estimator=True,
    )
    # extract the classifier of first pipeline from cv_results
    fitted_classifier = cv_results["estimator"][0][CLASSIFIER]

    return fitted_classifier, cv_results
