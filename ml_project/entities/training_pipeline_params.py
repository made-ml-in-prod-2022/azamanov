from dataclasses import dataclass

from omegaconf import DictConfig

from .splitting_params import SplittingParams
from .feature_params import FeatureParams
from .training_params import TrainingParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(stream: DictConfig) -> TrainingPipelineParams:
    schema = TrainingPipelineParamsSchema()
    return schema.load(yaml.safe_load(stream))
