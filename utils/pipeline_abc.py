import os
from abc import ABC, abstractmethod
import logging.config
from typing import Dict, List, Tuple, Union, Any

import plotly.express as px
import pandas as pd
import mlflow
from mlflow.exceptions import MlflowException
from plotly.graph_objs._figure import Figure
from sklearn.base import ClassifierMixin
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr

from utils.constants import (
    LOG_CONFIG_PATH,
    ARTIFACT_PATH,
    MLRUN_SQL_DATABASE_PATH,
    CLASSIFIER,
    X_COL,
    Y_COL,
    PIPELINE_HTML,
    SCORES_CSV,
    FEATURE_IMPORTANCE_PLOT,
)

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class ETLPipeline(ABC):
    _write_path: str
    _read_path: str

    @classmethod
    @abstractmethod
    def extract(cls):
        logger.info(f"Reading from {cls._read_path}")

    @classmethod
    @abstractmethod
    def transform(cls):
        logger.info("Transforming dataframe")

    @classmethod
    @abstractmethod
    def load(cls):
        logger.info(f"Writing to {cls._write_path}")

    @classmethod
    @abstractmethod
    def main(cls):
        cls.extract()
        cls.transform()
        cls.load()


class ModelingPipeline(ABC):
    # TODO add _method to specify that these methods should not be exposed

    @abstractmethod
    def __init__(
        self,
        experiment_name: str,
        read_path: str,
        X_col: List,
        y_col: str,
        params: Dict[str, Any],
        pipeline: Pipeline,
        scoring: Dict,
        tracking_uri: str,
        artifact_location: str,
    ):
        # TODO specify List[str] as type hint
        self._experiment_name = experiment_name
        self._read_path = read_path
        self._X_col = X_col
        self._y_col = y_col
        self._params = params
        self._pipeline = pipeline
        self._scoring = scoring
        self._tracking_uri = tracking_uri
        self._artifact_location = artifact_location
        self._fitted_classifier: ClassifierMixin
        self._X_train: pd.DataFrame
        self._y_train: pd.DataFrame

    @abstractmethod
    def load_data(self):
        logger.info(f"Loading data from {self._read_path}")
        X = pd.read_parquet(self._read_path, columns=self._X_col)
        y = pd.read_parquet(self._read_path, columns=[self._y_col])
        self._X_train, X_test, self._y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

    @abstractmethod
    def set_tags(self):
        logger.info("Setting tags")
        mlflow.set_tag(X_COL, self._X_col)
        mlflow.set_tag(Y_COL, self._y_col)

    @abstractmethod
    def log_params(self):
        logger.info("Logging params")
        mlflow.log_params(self._params)

    @abstractmethod
    def log_pipeline(self):
        logger.info("Logging pipeline artifact")
        with open(PIPELINE_HTML, "w") as f:
            f.write(estimator_html_repr(self._pipeline))
        mlflow.log_artifact(PIPELINE_HTML)
        os.remove(PIPELINE_HTML)

    @abstractmethod
    def evaluate_pipeline(self):
        logger.info("Evaluating pipeline")
        cross_validation = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        cv_results = cross_validate(
            self._pipeline,
            self._X_train,
            self._y_train,
            scoring=self._scoring,
            cv=cross_validation,
            n_jobs=-1,
            return_train_score=True,
            return_estimator=True,
        )

        # extract the classifier of first pipeline from cv_results
        self._fitted_classifier = cv_results["estimator"][0][CLASSIFIER]

        # remove estimator from cv_results dictionary to log metric and dataframe
        cv_results_without_estimator = {
            key: array for key, array in cv_results.items() if key != "estimator"
        }
        for key, array in cv_results_without_estimator.items():
            for i, value in enumerate(array):
                mlflow.log_metric(key, value, step=i + 1)
        self.log_df_artifact(pd.DataFrame(cv_results_without_estimator), SCORES_CSV)

    @abstractmethod
    def log_explainability(self):
        logger.info("Logging explainability")
        if hasattr(self._fitted_classifier, "feature_importances_"):
            feature_importance = pd.Series(
                data=self._fitted_classifier.feature_importances_,
                index=self._X_train.columns,
            ).sort_values()
            feature_importance_fig = px.bar(
                feature_importance,
                x=feature_importance.values,
                y=feature_importance.index,
                orientation="h",
                title="Feature Importance Plot",
            )
            self.log_plotly_artifact(feature_importance_fig, FEATURE_IMPORTANCE_PLOT)

    @abstractmethod
    def main(self):
        experiment_id = self.setup_mlflow()
        with mlflow.start_run(experiment_id=experiment_id) as active_run:
            self.load_data()
            self.set_tags()
            self.log_params()
            self.log_pipeline()
            self.evaluate_pipeline()
            self.log_explainability()

    def setup_mlflow(self):
        mlflow.set_tracking_uri(self._tracking_uri)
        try:
            experiment_id = mlflow.create_experiment(
                name=self._experiment_name, artifact_location=self._artifact_location
            )
        except MlflowException:
            experiment_id = mlflow.set_experiment(experiment_name=self._experiment_name)
        return experiment_id

    def log_df_artifact(self, df: pd.DataFrame, filename: str):
        df.to_csv(filename)
        mlflow.log_artifact(filename)
        os.remove(filename)

    def log_plotly_artifact(self, fig: Figure, filename: str):
        fig.write_html(filename)
        mlflow.log_artifact(filename)
        os.remove(filename)
