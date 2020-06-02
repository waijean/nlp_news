import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal
from sklearn.tree import DecisionTreeClassifier

from data_modeling.modeling import load_and_split_data, evaluate_cv_pipeline
from utils.constants import iris_X_COL, iris_y_COL, MICRO_CLASSIFIER_SCORING


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
        test_pipeline, expected_X_train, expected_y_train, MICRO_CLASSIFIER_SCORING
    )

    assert isinstance(fitted_classifier, DecisionTreeClassifier)

    expected_metrics = {
        key: array
        for key, array in expected_cv_result.items()
        if key not in ["estimator", "fit_time", "score_time"]
    }
    for key, value in expected_metrics.items():
        assert all(cv_results[key] == value)
