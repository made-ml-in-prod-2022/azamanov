import logging
import sys

import click

from data import read_data
from models import (
    load_model,
    save_predicts,
    predict_model
)
from entities import read_predict_pipeline_params


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(config_path: str):
    predict_pipeline_params = read_predict_pipeline_params(config_path)

    return run_predict_pipeline(predict_pipeline_params)


def run_predict_pipeline(predict_pipeline_params):
    logger.info(f"start predict pipeline with params {predict_pipeline_params}")
    data = read_data(predict_pipeline_params.dataset_path)
    logger.info(f"data.shape is {data.shape}")

    model = load_model(predict_pipeline_params.model_path)
    predicts = predict_model(model, data)

    save_predicts(predicts, predict_pipeline_params.output_prediction_path)


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()
