import pytest
import pandas as pd
import os
import pickle


@pytest.fixture
def input_true():
    return True

@pytest.fixture()
def pd_df():
    dataset_path = "../data/titanic/"
    train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
    return train_df

@pytest.fixture()
def model():
    model_path = "../saved_models/test.sav"
    model = pickle.load(open(model_path, mode="rb"))
    return model
