import mlflow
import pandas as pd
import numpy as np
from pandas._testing import assert_frame_equal

from utils.constants import FEATURE_IMPORTANCE_PLOT, TEST_EXPERIMENT_NAME, TEST_RUN_NAME


def test_log_single_run(
    setup_mlflow_run,
    grid_search_pipeline,
    full_params,
    test_metrics,
    test_fitted_classifier,
):
    grid_search_pipeline.log_single_run(
        full_params, test_metrics, test_fitted_classifier
    )
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(setup_mlflow_run.info.run_id).data
    assert data.params == {str(key): str(value) for key, value in full_params.items()}
    assert data.metrics == test_metrics

    artifacts = client.list_artifacts(setup_mlflow_run.info.run_id)
    artifacts_paths = [artifact.path for artifact in artifacts]
    assert FEATURE_IMPORTANCE_PLOT in artifacts_paths


def test_log_child_runs(
    setup_mlflow_run, grid_search_pipeline, setup_mlflow_experiment_id, expected_gs
):
    grid_search_pipeline.log_child_runs(
        setup_mlflow_experiment_id, expected_gs.cv_results_
    )
    query = f"tags.mlflow.parentRunId = '{setup_mlflow_run.info.run_id}'"
    runs_df_index = [
        "params.standardscaler__with_mean",
        "params.decisiontreeclassifier__criterion",
        "params.decisiontreeclassifier__max_depth",
    ]
    runs_df = mlflow.search_runs(
        experiment_ids=setup_mlflow_experiment_id, filter_string=query
    ).set_index(runs_df_index)
    runs_df = runs_df[
        [
            col
            for col in runs_df.columns
            if col.startswith("metrics") and not col.endswith("time")
        ]
    ]
    # remove metrics. from the column names to match expected_runs_df
    runs_df = runs_df.rename(lambda x: x[8:], axis="columns")

    expected_runs_df_index = runs_df_index = [
        "param_standardscaler__with_mean",
        "param_decisiontreeclassifier__criterion",
        "param_decisiontreeclassifier__max_depth",
    ]
    expected_runs_df = pd.DataFrame(expected_gs.cv_results_)
    # convert values in param columns to string before setting as index
    for col in expected_runs_df_index:
        expected_runs_df[col] = expected_runs_df[col].astype("str")
    expected_runs_df = expected_runs_df.set_index(expected_runs_df_index)
    # rename the index to match runs df
    expected_runs_df.index = expected_runs_df.index.rename(runs_df_index)
    expected_runs_df = expected_runs_df[
        [
            col
            for col in expected_runs_df.columns
            if (col.startswith("mean_") or col.startswith("std_"))
            and not col.endswith("time")
        ]
    ]
    assert_frame_equal(runs_df, expected_runs_df, check_like=True)


def test_main(grid_search_pipeline):
    grid_search_pipeline.main()
    experiment_id = mlflow.get_experiment_by_name(TEST_EXPERIMENT_NAME).experiment_id
    df = mlflow.search_runs(
        experiment_ids=experiment_id,
        filter_string=f"tags.mlflow.runName = '{TEST_RUN_NAME}'",
    )
    assert all(df["status"].values == "FINISHED")
