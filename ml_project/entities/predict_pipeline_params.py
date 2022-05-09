from dataclasses import dataclass

from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictPipelineParams:
    model_path: str
    dataset_path: str
    output_prediction_path: str


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
