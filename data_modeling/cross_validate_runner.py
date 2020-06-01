from typing import Dict, List, Union, Tuple, Any

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from data_modeling.data_preprocessing.vectorize import SpacyVectorizer
from utils.constants import (
    CLEANED_NEWS_TITLE_PATH,
    COL_TITLE,
    COL_SIGN,
    MLRUN_SQL_DATABASE_PATH,
    ARTIFACT_PATH,
    VECTORIZER,
    CLASSIFIER,
    CLASSIFIER_SCORING,
)
from utils.pipeline_abc import ModelingPipeline


class CrossValidatePipeline(ModelingPipeline):
    def __init__(
        self,
        experiment_name: str,
        read_path: str,
        X_col: List,
        y_col: str,
        params: Dict[str, Any],
        pipeline: Pipeline,
        scoring: Dict = CLASSIFIER_SCORING,
        tracking_uri: str = MLRUN_SQL_DATABASE_PATH,
        artifact_location: str = ARTIFACT_PATH,
    ):
        """
        Create a cross- validation pipeline to train and evaluate a specific pipeline

        Args:
            experiment_name: the experiment name for the grouping of runs
            read_path: path to parquet file which contains both feature and target columns
            X_col: list of feature columns
            y_col: target column
            params: parameters to log
            pipeline: sklearn Pipeline which can contain a series of estimator. It must have a classifier for the last step
            tracking_uri: location to store run details such as params, tags and metrics
            artifact_location: location to store artifacts such as html representation of pipeline, plots and models
        """
        super().__init__(
            experiment_name,
            read_path,
            X_col,
            y_col,
            params,
            pipeline,
            scoring,
            tracking_uri,
            artifact_location,
        )

    def load_data(self):
        super().load_data()

    def set_tags(self):
        super().set_tags()

    def log_params(self):
        super().log_params()

    def log_pipeline(self):
        super().log_pipeline()

    def evaluate_pipeline(self):
        super().evaluate_pipeline()

    def log_explainability(self):
        super().log_explainability()

    def main(self):
        super().main()


if __name__ == "__main__":
    # create params
    vect_param = {"ngram_range": (1, 1)}
    clf_param = {"kernel": "linear"}
    params: Dict[str, Any] = {**vect_param, **clf_param}

    # create estimator and pipeline
    vect = SpacyVectorizer(**vect_param)
    classifier = SVC(**clf_param)
    pipeline = Pipeline([(VECTORIZER, vect), (CLASSIFIER, classifier)])

    CrossValidatePipeline(
        experiment_name="SpacyVectorizer SVC",
        read_path=CLEANED_NEWS_TITLE_PATH,
        X_col=[COL_TITLE],
        y_col=COL_SIGN,
        params=params,
        pipeline=pipeline,
    ).main()
