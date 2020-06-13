from dataclasses import dataclass, field
from typing import Dict, List, Any, Union

import mlflow
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC

from data_modeling.data_preprocessing.vectorize import SpacyVectorizer
from data_modeling.mlrun import (
    set_tags,
    log_params,
    log_pipeline,
    log_explainability,
    setup_mlflow,
    log_metrics,
)
from data_modeling.scoring import CLASSIFIER_SCORING
from utils.constants import (
    CLEANED_NEWS_TITLE_PATH,
    COL_TITLE,
    COL_SIGN,
    MLRUN_SQL_DATABASE_PATH,
    ARTIFACT_PATH,
    VECTORIZER,
    CLASSIFIER,
    ACCURACY,
    DEFAULT_CV,
)
from data_modeling.modeling import (
    load_and_split_data,
    evaluate_cv_pipeline,
    evaluate_grid_search_pipeline,
)


@dataclass
class GridSearchPipeline:
    """
    Create a cross- validation pipeline to train and evaluate a specific pipeline

    Args:
        experiment_name: the experiment name for the grouping of runs
        read_path: path to parquet file which contains both feature and target columns
        X_col: list of feature columns
        y_col: target column
        params_grid: grid parameters to perform grid search
        pipeline: sklearn Pipeline which can contain a series of estimator. It must have a classifier for the last step
        tracking_uri: location to store run details such as params, tags and metrics
        artifact_location: location to store artifacts such as html representation of pipeline, plots and models
    """

    experiment_name: str
    run_name: str
    read_path: str
    X_col: List
    y_col: str
    param_grid: Dict[str, Any]
    pipeline: Pipeline
    refit: str
    scoring: Dict
    cv = DEFAULT_CV
    tracking_uri: str = MLRUN_SQL_DATABASE_PATH
    artifact_location: str = ARTIFACT_PATH

    def main(self):
        experiment_id = setup_mlflow(
            self.tracking_uri, self.experiment_name, self.artifact_location
        )
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=self.run_name
        ) as parent_run:
            X_train, X_test, y_train, y_test = load_and_split_data(
                self.read_path, self.X_col, self.y_col
            )
            set_tags(self.X_col, self.y_col)
            log_pipeline(self.pipeline)
            best, cv_results = evaluate_grid_search_pipeline(
                self.pipeline,
                self.param_grid,
                X_train,
                y_train,
                self.scoring,
                self.cv,
                self.refit,
            )
            # TODO log best model
            self.log_single_run(**best)
            self.log_child_runs(experiment_id, cv_results)

    def log_child_runs(self, experiment_id, cv_results):
        for i in range(len(cv_results["params"])):
            param_dict = {
                key[6:]: value[i]
                for key, value in cv_results.items()
                if key.startswith("param_")
            }
            metrics_dict = {
                key: value[i]
                for key, value in cv_results.items()
                if key.startswith("mean_") or key.startswith("std_")
            }
            with mlflow.start_run(
                experiment_id=experiment_id, nested=True
            ) as child_run:
                # TODO specify run name for child run
                self.log_single_run(param_dict, metrics_dict)

    def log_single_run(
        self, params: Dict[str, Any], metrics: Dict[str, Any], fitted_classifier=None
    ):
        log_params(params)
        log_metrics(metrics)
        if fitted_classifier is not None:
            log_explainability(fitted_classifier, self.X_col)


if __name__ == "__main__":
    # create params grid
    param_grid = {
        "vectorizer__min_df": [1, 2],
        "vectorizer__max_df": [0.7, 0.8],
        "classifier__C": [0.1, 1, 10],
    }

    # create estimator and pipeline
    vect = SpacyVectorizer()
    classifier = SVC()
    pipeline = make_pipeline(vect, classifier)

    GridSearchPipeline(
        experiment_name="Grid Search",
        run_name="SpacyVectorizer SVC",
        read_path=CLEANED_NEWS_TITLE_PATH,
        X_col=[COL_TITLE],
        y_col=COL_SIGN,
        param_grid=param_grid,
        pipeline=pipeline,
        refit=ACCURACY,
        scoring=CLASSIFIER_SCORING,
    ).main()
