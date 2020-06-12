import mlflow
import numpy as np

from data_modeling.mlrun import (
    set_tags,
    log_params,
    log_pipeline,
    log_explainability,
    log_df_artifact,
    log_cv_metrics,
    log_metrics,
    get_params,
)
from utils.constants import (
    X_COL,
    Y_COL,
    iris_X_COL,
    iris_y_COL,
    PIPELINE_HTML,
    FEATURE_IMPORTANCE_PLOT,
    SCORES_CSV,
    FEATURE_IMPORTANCE_CSV,
)


def test_set_tags(setup_mlflow_run):
    set_tags(iris_X_COL, iris_y_COL)
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(setup_mlflow_run.info.run_id).data
    # assert individual tags because the tags property in RunData contains additional tags (mlflow.source.name, mlflow.source.type, mlflow.user)
    assert data.tags[X_COL] == f"{iris_X_COL}"
    assert data.tags[Y_COL] == f"{iris_y_COL}"


def test_get_params(test_pipeline, full_params):
    actual_full_params = get_params(test_pipeline)
    actual_full_params_str = {
        str(key): str(value) for key, value in actual_full_params.items()
    }
    full_params_str = {str(key): str(value) for key, value in full_params.items()}
    assert actual_full_params_str == full_params_str


def test_log_params(setup_mlflow_run, full_params):
    log_params(full_params)
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(setup_mlflow_run.info.run_id).data
    # the params property in RunData is a dictionary with keys and values as string
    assert data.params == {str(key): str(value) for key, value in full_params.items()}


def test_log_metrics(setup_mlflow_run, test_metrics):
    log_metrics(test_metrics)
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(setup_mlflow_run.info.run_id).data
    assert data.metrics == test_metrics


def test_log_pipeline(setup_mlflow_run, test_pipeline):
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
    log_pipeline(test_pipeline)
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(setup_mlflow_run.info.run_id)
    artifacts_paths = [artifact.path for artifact in artifacts]
    assert PIPELINE_HTML in artifacts_paths


def test_log_explainability(setup_mlflow_run, test_fitted_classifier, expected_X_train):
    log_explainability(test_fitted_classifier, iris_X_COL)
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(setup_mlflow_run.info.run_id)
    artifacts_paths = [artifact.path for artifact in artifacts]
    assert FEATURE_IMPORTANCE_PLOT in artifacts_paths
    assert FEATURE_IMPORTANCE_CSV in artifacts_paths


def test_log_df_artifact(setup_mlflow_run, expected_X_train):
    # TODO read dataframe and assert df
    log_df_artifact(expected_X_train, SCORES_CSV)
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(setup_mlflow_run.info.run_id)
    artifacts_paths = [artifact.path for artifact in artifacts]
    assert SCORES_CSV in artifacts_paths


def test_log_cv_metrics(setup_mlflow_run, expected_cv_result):
    log_cv_metrics(expected_cv_result)
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(setup_mlflow_run.info.run_id).data
    expected_metrics = {
        key: np.mean(array)
        for key, array in expected_cv_result.items()
        if key not in ["estimator", "fit_time", "score_time"]
    }
    for key, value in expected_metrics.items():
        assert data.metrics[key] == value
