"""Test file for utils.py"""
import pytest
from pandas import DataFrame

from utils import DataPreprocessor

@pytest.fixture(name="data_preprocessor")
def data_pp():
    """
    Returns:
        DataPreprocessor: instance of class
    """
    return DataPreprocessor()


def test_dp_init(data_preprocessor):
    """Test DataPreprocessor constructor"""
    assert isinstance(data_preprocessor, DataPreprocessor)

def test_df(pd_df):
    """Check against expected train dataset shape"""
    assert pd_df.shape == (891, 12)

def test_dp_na(pd_df: DataFrame, data_preprocessor):
    """Test: there shouldn't be NAN values"""
    x_data, y_data = data_preprocessor.preprocess_dataset(pd_df)
    assert not x_data.isna().values.any()
    assert not y_data.isna().values.any()

    x_data = data_preprocessor.preprocess_dataset(pd_df, test=True)
    assert not x_data.isna().values.any()

def test_dp_target(pd_df: DataFrame, data_preprocessor):
    """Test: dataset target 'Survived' should be in X part of data"""
    x_data = data_preprocessor.preprocess_dataset(pd_df, test=True)
    assert 'Survived' not in x_data.columns
