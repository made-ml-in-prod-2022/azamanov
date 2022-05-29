import pickle

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def load_model(model_path: str) -> Pipeline:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def save_predicts(predicts: pd.Series, out_path: str) -> None:
    with open(out_path, "w+") as f:
        predicts_str = ",\n".join(predicts.astype(str))
        f.write(predicts_str)
