import logging.config
import os
from typing import Dict, Any, List
import pandas as pd

import mlflow
from mlflow.exceptions import MlflowException
from plotly.graph_objs._figure import Figure
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr
from plotly import express as px

from utils.constants import (
    LOG_CONFIG_PATH,
    X_COL,
    Y_COL,
    PIPELINE_HTML,
    SCORES_CSV,
    FEATURE_IMPORTANCE_PLOT,
)

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str, experiment_name: str, artifact_location: str):
    mlflow.set_tracking_uri(tracking_uri)
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location
        )
    except MlflowException:
        experiment_id = mlflow.set_experiment(experiment_name=experiment_name)
    return experiment_id


def set_tags(X_col: List[str], y_col: str):
    logger.info("Setting tags")
    mlflow.set_tag(X_COL, X_col)
    mlflow.set_tag(Y_COL, y_col)


def log_params(params: Dict[str, Any]):
    logger.info("Logging params")
    mlflow.log_params(params)


def log_pipeline(pipeline: Pipeline):
    logger.info("Logging pipeline artifact")
    with open(PIPELINE_HTML, "w") as f:
        f.write(estimator_html_repr(pipeline))
    mlflow.log_artifact(PIPELINE_HTML)
    os.remove(PIPELINE_HTML)


def log_metrics(cv_results: Dict[str, Any]):
    # remove estimator from cv_results dictionary to log metric and dataframe
    cv_results_without_estimator = {
        key: array for key, array in cv_results.items() if key != "estimator"
    }
    for key, array in cv_results_without_estimator.items():
        for i, value in enumerate(array):
            mlflow.log_metric(key, value, step=i + 1)
    log_df_artifact(pd.DataFrame(cv_results_without_estimator), SCORES_CSV)


def log_df_artifact(df: pd.DataFrame, filename: str):
    df.to_csv(filename)
    mlflow.log_artifact(filename)
    os.remove(filename)


def log_explainability(fitted_classifier, X_train):
    logger.info("Logging explainability")
    if hasattr(fitted_classifier, "feature_importances_"):
        feature_importance = pd.Series(
            data=fitted_classifier.feature_importances_, index=X_train.columns,
        ).sort_values()
        feature_importance_fig = px.bar(
            feature_importance,
            x=feature_importance.values,
            y=feature_importance.index,
            orientation="h",
            title="Feature Importance Plot",
        )
        log_plotly_artifact(feature_importance_fig, FEATURE_IMPORTANCE_PLOT)


def log_plotly_artifact(fig: Figure, filename: str):
    fig.write_html(filename)
    mlflow.log_artifact(filename)
    os.remove(filename)
