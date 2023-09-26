import pytest
import pandas as pd
import os
import pickle

from pathlib import Path
data_path = (Path(__file__).parent.parent / "data").absolute()
model_path = (Path(__file__).parent.parent / "saved_models").absolute()


@pytest.fixture
def input_true():
    return True

@pytest.fixture()
def pd_df():
    path = os.path.join(data_path, "titanic")
    train_df = pd.read_csv(os.path.join(path, "train.csv"))
    return train_df

@pytest.fixture()
def model():
    path = os.path.join(model_path, "test.sav")
    model = pickle.load(open(path, mode="rb"))
    return model
