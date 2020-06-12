from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC

from data_modeling.data_preprocessing.vectorize import SpacyVectorizer
from data_modeling.mlrun import (
    set_tags,
    log_params,
    log_pipeline,
    log_cv_metrics,
    log_explainability,
    setup_mlflow,
    get_params,
)
from utils.constants import (
    CLEANED_NEWS_TITLE_PATH,
    COL_TITLE,
    COL_SIGN,
    MLRUN_SQL_DATABASE_PATH,
    ARTIFACT_PATH,
    VECTORIZER,
    CLASSIFIER,
    CLASSIFIER_SCORING,
    DEFAULT_CV,
)
from data_modeling.modeling import load_and_split_data, evaluate_cv_pipeline


@dataclass
class CrossValidatePipeline:
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

    experiment_name: str
    run_name: str
    read_path: str
    X_col: List
    y_col: str
    pipeline: Pipeline
    scoring: Dict
    params: Optional[Dict[str, Any]] = None
    cv = DEFAULT_CV
    tracking_uri: str = MLRUN_SQL_DATABASE_PATH
    artifact_location: str = ARTIFACT_PATH

    def main(self):
        experiment_id = setup_mlflow(
            self.tracking_uri, self.experiment_name, self.artifact_location
        )
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=self.run_name
        ) as active_run:
            X_train, X_test, y_train, y_test = load_and_split_data(
                self.read_path, self.X_col, self.y_col
            )
            set_tags(self.X_col, self.y_col)
            if self.params is None:
                self.params = get_params(self.pipeline)
            log_params(self.params)
            log_pipeline(self.pipeline)
            fitted_classifier, cv_results = evaluate_cv_pipeline(
                self.pipeline, X_train, y_train, self.scoring, self.cv
            )
            log_cv_metrics(cv_results)
            log_explainability(fitted_classifier, self.X_col)


if __name__ == "__main__":
    # create estimator and pipeline
    vect = SpacyVectorizer(ngram_range=(1, 1))
    classifier = SVC(kernel="linear")
    pipeline = make_pipeline(vect, classifier)

    CrossValidatePipeline(
        experiment_name="Cross Validate",
        run_name="SpacyVectorizer SVC",
        read_path=CLEANED_NEWS_TITLE_PATH,
        X_col=[COL_TITLE],
        y_col=COL_SIGN,
        scoring=CLASSIFIER_SCORING,
        pipeline=pipeline,
    ).main()
