import codecs
import os

import mlflow
import pandas as pd
from mlflow.entities import FileInfo
from pandas.testing import assert_series_equal, assert_frame_equal
from sklearn.tree import DecisionTreeClassifier

from utils.constants import (
    X_COL,
    Y_COL,
    iris_X_COL,
    iris_y_COL,
    FULL_PARAM,
    PIPELINE_HTML,
    CLASSIFIER,
    SCORES_CSV,
    FEATURE_IMPORTANCE_PLOT,
    TEST_EXPERIMENT_NAME,
)


def test_main(cross_validate_pipeline):
    """
    Note: This has to run first before testing the individual methods within main(). Otherwise, we need to call mlflow.end_run() at the beginning and check all status is FINISHED at the end

    Setting experiment_ids as None will default to the active experiment
    """
    cross_validate_pipeline.main()
    experiment_id = mlflow.get_experiment_by_name(TEST_EXPERIMENT_NAME).experiment_id
    df = mlflow.search_runs(experiment_ids=experiment_id)
    assert df["status"].values == "FINISHED"


def test_load_data_from_tmp_dir(data_dir, iris_df):
    p = data_dir / "iris.parquet"
    actual_df = pd.read_parquet(p)
    assert actual_df.equals(iris_df)


def test_load_data(
    setup_mlflow_run, cross_validate_pipeline, expected_X_train, expected_y_train
):
    """
    https://www.mlflow.org/docs/latest/tracking.html#logging-functions

    Cannot access currently-active run attributes (parameters, metrics, etc.) through the run returned by mlflow.active_run
    In order to access such attributes, use the mlflow.tracking.MlflowClient as follows:
        client = mlflow.tracking.MlflowClient()
        data = client.get_run(mlflow.active_run().info.run_id).data
    """
    cross_validate_pipeline.load_data()
    assert_frame_equal(
        cross_validate_pipeline._X_train, expected_X_train, check_less_precise=True
    )
    assert_series_equal(
        cross_validate_pipeline._y_train["target"],
        expected_y_train,
        check_less_precise=True,
    )


def test_set_tags(setup_mlflow_run, cross_validate_pipeline):
    cross_validate_pipeline.set_tags()
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(setup_mlflow_run.info.run_id).data
    # assert individual tags because the tags property in RunData contains additional tags (mlflow.source.name, mlflow.source.type, mlflow.user)
    assert data.tags[X_COL] == f"{iris_X_COL}"
    assert data.tags[Y_COL] == f"{iris_y_COL}"


def test_log_params(setup_mlflow_run, cross_validate_pipeline):
    cross_validate_pipeline.log_params()
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(setup_mlflow_run.info.run_id).data
    # the params property in RunData is a dictionary with keys and values as string
    assert data.params == {str(key): str(value) for key, value in FULL_PARAM.items()}


def test_log_pipeline(setup_mlflow_run, cross_validate_pipeline):
    """
    Two ways of testing:
    1. Use the list artifacts method from MlflowClient and assert file path exists in that list
    2. Read the html file from artifact uri and assert the html file with the expected html file

    Note that the artifact uri is different from the artifact location we passed to CrossValidatePipeline
    because the artifact uri includes run id directory

    Method 2
    info = client.get_run(setup_mlflow_run.info.run_id).info
    path = os.path.join(info.artifact_uri, PIPELINE_HTML)
    f = codecs.open(path, 'r')
    f.read()
    """
    cross_validate_pipeline.log_pipeline()
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(setup_mlflow_run.info.run_id)
    artifacts_paths = [artifact.path for artifact in artifacts]
    assert PIPELINE_HTML in artifacts_paths


def test_evaluate_pipeline_and_log_explainability(
    setup_mlflow_run, cross_validate_pipeline, cv_result
):
    """
    Note:
    - Need to load data before training model
    - Combine evaluate pipeline and log explainability so don't have to train the pipeline twice

    1. Assert the fitted estimator is a DecisionTreeClassifier
    2. Check the metrics property in RunData (excluding certain keys such as "estimator", "fit_time" and "score_time")
    3. Assert file path exists in artifacts list
    #TODO read dataframe and assert df
    """
    cross_validate_pipeline.load_data()
    cross_validate_pipeline.evaluate_pipeline()
    cross_validate_pipeline.log_explainability()

    assert isinstance(
        cross_validate_pipeline._fitted_classifier, DecisionTreeClassifier
    )

    client = mlflow.tracking.MlflowClient()
    data = client.get_run(setup_mlflow_run.info.run_id).data
    expected_metrics = {
        key: max(array)
        for key, array in cv_result.items()
        if key not in ["estimator", "fit_time", "score_time"]
    }
    for key, value in expected_metrics.items():
        assert data.metrics[key] == value

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(setup_mlflow_run.info.run_id)
    artifacts_paths = [artifact.path for artifact in artifacts]
    assert SCORES_CSV in artifacts_paths

    assert FEATURE_IMPORTANCE_PLOT in artifacts_paths
