# -*- coding: utf-8 -*-
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from ml_project.entities import SplittingParams

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    logger.debug("Read dataset from %s, len = %d".format(path, len(data)))
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(
        data, test_size=params.test_size, random_state=params.random_state
    )
    logger.debug("Split dataset to train and test")
    return train_data, test_data
