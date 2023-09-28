"""Titanic train/test script"""
import os
import uuid
import pickle

import mlflow
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.models import infer_signature

from utils import k_cross_val, DataPreprocessor
import rftuner as rft


MLFLOW_TRACKING_URI = "sqlite:///../mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = mlflow.tracking.MlflowClient()
DATASET_PATH = "./data/titanic/"
train_df = pd.read_csv(os.path.join(DATASET_PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(DATASET_PATH, "test.csv"))


mlflow_exp_name = "titanic-hyp-" + str(uuid.uuid4()).split("-", maxsplit=1)[0]
preprocessor = DataPreprocessor()

best_params = rft.tune_model(train_df,
                         n_trials=100,
                         preprocessor=preprocessor,
                         study_name=mlflow_exp_name,
                         mlflow_tracking_uri=MLFLOW_TRACKING_URI)

mlflow.set_experiment("titanic")
mlflow.sklearn.autolog(disable=True)
with mlflow.start_run(run_name='rf_baseline'):
    mlflow.set_tag("model_name", "RF")

    model = RandomForestClassifier(**best_params)
    param = model.get_params()

    X_train, y_train = preprocessor.preprocess_dataset(train_df)

    model.fit(X_train, np.ravel(y_train))

    acc = k_cross_val(k = 5,
                    model_class = RandomForestClassifier,
                    model_params = param,
                    data = train_df,
                    preprocessor=preprocessor)

    joined_train = pd.concat((X_train,y_train),axis=1)
    DB_PATH = "../data/df.parquet.gzip"
    joined_train.to_parquet(DB_PATH, compression="gzip")
    mlflow.log_artifact(DB_PATH)
    mlflow_train_dataset: PandasDataset = mlflow.data.from_pandas(joined_train)
    mlflow.log_input(mlflow_train_dataset, context="training")
    mlflow.log_params(params=param)
    mlflow.log_metric("accuracy", acc)
    signature = infer_signature(model_input=X_train, model_output=y_train)
    mlflow.sklearn.log_model(model, "sk_models", signature=signature)

MODEL_FILENAME = "../saved_models/test.sav"
with open(MODEL_FILENAME, mode="wb") as file:
    pickle.dump(model, file)

X_test = preprocessor.preprocess_dataset(test_df, test=True)

predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_df.PassengerId,
                    'Survived': predictions})
output.to_csv('../kaggle/titanic/submission.csv', index=False)
print("Your submission was successfully saved!")
