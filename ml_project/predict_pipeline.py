import logging
import sys
from typing import Dict

import hydra

from ml_project.dataset import read_data, get_df_from_data
from ml_project.models import load_model, save_predicts, predict_model
from ml_project.entities import read_predict_pipeline_params, PredictPipelineParams
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(config_path: str) -> None:
    predict_pipeline_params = read_predict_pipeline_params(config_path)

    run_predict_pipeline(predict_pipeline_params)


def run_predict_pipeline(predict_pipeline_params: PredictPipelineParams) -> None:
    logger.info(f"start predict pipeline with params {predict_pipeline_params}")
    data = read_data(predict_pipeline_params.dataset_path)
    logger.info(f"data.shape is {data.shape}")

    model = load_model(predict_pipeline_params.model_path)
    predicts = predict_model(model, data)

    save_predicts(predicts, predict_pipeline_params.output_prediction_path)


def online_predict_pipeline(data: Dict, model: Pipeline):
    logger.info(f"start predict pipeline with {data}")
    df = get_df_from_data(data)
    logger.info("get df from data")
    predicts = predict_model(model, df)
    logger.info(f"predict: {predicts}")
    return predicts


@hydra.main(config_path="../configs", config_name="pred")
def predict_pipeline_command(config_path: str) -> None:
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()
