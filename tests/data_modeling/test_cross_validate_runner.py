import mlflow

from utils.constants import TEST_EXPERIMENT_NAME


def test_main(cross_validate_pipeline):
    """
    Note: This has to run first before testing the individual methods within main(). Otherwise, we need to call mlflow.end_run() at the beginning and check all status is FINISHED at the end

    Setting experiment_ids as None will default to the active experiment
    """
    cross_validate_pipeline.main()
    experiment_id = mlflow.get_experiment_by_name(TEST_EXPERIMENT_NAME).experiment_id
    df = mlflow.search_runs(experiment_ids=experiment_id)
    assert df["status"].values == "FINISHED"
