import os
from typing import List
from py._path.local import LocalPath

from ml_project.train_pipeline import run_train_pipeline
from ml_project.predict_pipeline import run_predict_pipeline
from ml_project.entities import (
    TrainingPipelineParams,
    PredictPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
)


def test_train_e2e(
    tmpdir: LocalPath,
    dataset_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
) -> None:
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    expected_output_prediction_path = tmpdir.join("predictions.txt")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(test_size=0.2, random_state=239),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
        ),
        train_params=TrainingParams(model_type="RandomForestClassifier"),
    )
    real_model_path, metrics = run_train_pipeline(params)
    assert metrics["roc_auc_score"] > 0
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
    pred_params = PredictPipelineParams(
        model_path=real_model_path,
        dataset_path=dataset_path,
        output_prediction_path=expected_output_prediction_path,
    )
    run_predict_pipeline(pred_params)
    assert os.path.exists(expected_output_model_path)
