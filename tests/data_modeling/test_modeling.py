import pandas as pd
import numpy as np
from pandas._testing import assert_frame_equal, assert_series_equal
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data_modeling.modeling import (
    load_and_split_data,
    evaluate_cv_pipeline,
    evaluate_grid_search_pipeline,
)
from utils.constants import (
    iris_X_COL,
    iris_y_COL,
    MICRO_CLASSIFIER_SCORING,
    DEFAULT_CV,
    ACCURACY,
)


def test_load_data_from_tmp_dir(data_dir, iris_df):
    p = data_dir / "iris.parquet"
    actual_df = pd.read_parquet(p)
    assert actual_df.equals(iris_df)


def test_load_and_split_data(data_dir, expected_X_train, expected_y_train):
    """
    https://www.mlflow.org/docs/latest/tracking.html#logging-functions

    Cannot access currently-active run attributes (parameters, metrics, etc.) through the run returned by mlflow.active_run
    In order to access such attributes, use the mlflow.tracking.MlflowClient as follows:
        client = mlflow.tracking.MlflowClient()
        data = client.get_run(mlflow.active_run().info.run_id).data
    """
    read_path = data_dir / "iris.parquet"
    X_train, X_test, y_train, y_test = load_and_split_data(
        read_path, iris_X_COL, iris_y_COL
    )
    assert_frame_equal(X_train, expected_X_train, check_less_precise=True)
    assert_series_equal(
        y_train["target"], expected_y_train, check_less_precise=True,
    )


def test_evaluate_cv_pipeline(
    setup_mlflow_run,
    test_pipeline,
    expected_X_train,
    expected_y_train,
    expected_cv_result,
):
    """
    1. Assert the fitted estimator is a DecisionTreeClassifier
    2. Check the metrics property in RunData (excluding certain keys such as "estimator", "fit_time" and "score_time")
    """
    fitted_classifier, cv_results = evaluate_cv_pipeline(
        test_pipeline,
        expected_X_train,
        expected_y_train,
        MICRO_CLASSIFIER_SCORING,
        DEFAULT_CV,
    )

    assert isinstance(fitted_classifier, DecisionTreeClassifier)

    expected_metrics = {
        key: array
        for key, array in expected_cv_result.items()
        if key not in ["estimator", "fit_time", "score_time"]
    }
    for key, value in expected_metrics.items():
        assert all(cv_results[key] == value)


def test_evaluate_grid_search_pipeline(
    test_pipeline, test_param_grid, expected_X_train, expected_y_train, expected_gs
):
    best, cv_results = evaluate_grid_search_pipeline(
        test_pipeline,
        test_param_grid,
        expected_X_train,
        expected_y_train,
        scoring=MICRO_CLASSIFIER_SCORING,
        cv=DEFAULT_CV,
        refit=ACCURACY,
    )
    for key, value in cv_results.items():
        if key.startswith("param_"):
            assert all(value == expected_gs.cv_results_[key])
        if (key.startswith("mean_") or key.startswith("std_")) and not key.endswith(
            "time"
        ):
            assert np.allclose(value, expected_gs.cv_results_[key])

    assert best["params"] == expected_gs.best_params_
    assert np.isclose(best["metrics"][ACCURACY], expected_gs.best_score_)
    assert isinstance(best["fitted_classifier"], Pipeline)
