import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import mlflow
from sklearn.ensemble import RandomForestClassifier
import sklearn

from sklearn.metrics import accuracy_score
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.models import infer_signature

from utils import tune_model, kCrossVal, preprocess_dataset
import uuid


mlflow.set_tracking_uri("sqlite:///../mlflow.db")
      
client = mlflow.tracking.MlflowClient()

dataset_path = "../data/titanic/"
train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
test_df = pd.read_csv(os.path.join(dataset_path, "test.csv"))



mlflow_exp_name = "titanic-hyp-" + str(uuid.uuid4()).split("-")[0]
best_params = tune_model(train_df, n_trials=100, mlflow_exp_name=mlflow_exp_name)

mlflow.set_experiment("titanic")
mlflow.sklearn.autolog(disable=True)
with mlflow.start_run(run_name='rf_baseline'):
    mlflow.set_tag("model_name", "RF")    

    model = RandomForestClassifier(**best_params)
    param = model.get_params()
    
    acc = kCrossVal(k = 5, 
                    model_class = RandomForestClassifier, 
                    model_params = param, 
                    data = train_df)

    X_train, y_train = preprocess_dataset(train_df)
    model.fit(X_train, np.ravel(y_train))

    joined_train = pd.concat((X_train,y_train),axis=1)
    mlflow_train_dataset: PandasDataset = mlflow.data.from_pandas(joined_train)
    mlflow.log_input(mlflow_train_dataset, context="training")
    mlflow.log_params(params=param)
    mlflow.log_metric("accuracy", acc)
    signature = infer_signature(model_input=X_train, model_output=y_train)
    mlflow.sklearn.log_model(model, "sk_models", signature=signature)

    print(acc)
    
X_test = preprocess_dataset(test_df, test=True)

predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_df.PassengerId,
                    'Survived': predictions})
output.to_csv('../kaggle/titanic/submission.csv', index=False)
print("Your submission was successfully saved!")