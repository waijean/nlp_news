import mlflow

from utils.constants import TEST_EXPERIMENT_NAME


def test_main(cross_validate_pipeline):
    cross_validate_pipeline.main()
    experiment_id = mlflow.get_experiment_by_name(TEST_EXPERIMENT_NAME).experiment_id
    df = mlflow.search_runs(experiment_ids=experiment_id)
    assert df["status"].values == "FINISHED"
