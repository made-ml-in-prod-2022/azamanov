import pickle
from typing import Union, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from entities import TrainingParams

SklearnClassificationModel = Union[RandomForestClassifier, KNeighborsClassifier]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassificationModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators, random_state=train_params.random_state
        )
    elif train_params.model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(
            n_neighbors=train_params.n_neighbors
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def evaluate_model(
    predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    return {
        "roc_auc_score": roc_auc_score(target, predicts),
        "f1_score": f1_score(target, predicts),
        "accuracy_score": accuracy_score(target, predicts),
    }


def create_inference_pipeline(
    model: SklearnClassificationModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
