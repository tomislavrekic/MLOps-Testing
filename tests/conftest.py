"""conftest file, fixtures for testing"""
import os
import pickle
from pathlib import Path
import pytest
import pandas as pd
from pandas import DataFrame


data_path = (Path(__file__).parent.parent / "data").absolute()
model_path = (Path(__file__).parent.parent / "saved_models").absolute()

@pytest.fixture()
def pd_df() -> DataFrame:
    """
    Returns:
        DataFrame : train dataset
    """
    path = os.path.join(data_path, "titanic")
    train_df = pd.read_csv(os.path.join(path, "train.csv"))
    return train_df

@pytest.fixture()
def model():
    """
    Returns sklearn model
    """
    path = os.path.join(model_path, "test.sav")
    with open(path, mode="rb") as file:
        return pickle.load(file)
