import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / "src"
print(src_path)
src_path = src_path.absolute()
print(src_path)
sys.path.append(src_path)
from utils import DataPreprocessor, kCrossVal
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
import pytest

@pytest.fixture()
def dp():
    return DataPreprocessor()


def test_dp_init(dp):
    assert isinstance(dp, DataPreprocessor)

def test_df(pd_df):
    assert (pd_df.shape == (891, 12))


def test_fixture(input_true):
    assert input_true

def test_dp_na(pd_df: DataFrame, dp):
    X, Y = dp.preprocess_dataset(pd_df)
    assert not X.isna().values.any()
    assert not Y.isna().values.any()

    X = dp.preprocess_dataset(pd_df, test=True)
    assert not X.isna().values.any()

def test_dp_target(pd_df: DataFrame, dp):
    X = dp.preprocess_dataset(pd_df, test=True)
    assert 'Survived' not in X.columns

