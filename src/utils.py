import pandas as pd
import os
import numpy as np

import mlflow
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

client = mlflow.tracking.MlflowClient()

titles_map = {
    'Mr' :         'Mr',
    'Mme':         'Mrs',
    'Ms':          'Mrs',
    'Mrs' :        'Mrs',
    'Master' :     'Master',
    'Mlle':        'Miss',
    'Miss' :       'Miss',
    'Capt':        'Officer',
    'Col':         'Officer',
    'Major':       'Officer',
    'Dr':          'Officer',
    'Rev':         'Officer',
    'Jonkheer':    'Royalty',
    'Don':         'Royalty',
    'Sir' :        'Royalty',
    'Countess':    'Royalty',
    'Dona':        'Royalty',
    'Lady' :       'Royalty'
}

class DataPreprocessor():
    def __init__(self, data) -> None:
        self.col_index_set = False
        self.data = data
        
        #in_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Age', 'Embarked']
        #out_features = ['Survived']
        pass

    def extract_title(self, names):
        '''Extracts the title from the passenger names.'''

        return names.str.extract(' ([A-Za-z]+)\.', expand=False).map(titles_map)

    def preprocess_dataset(self, df, test=False):
        self.in_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Age', 'Embarked']
        self.out_features = ['Survived']

        #train_df['Title'] = train_df['Name'].map(lambda x: x.split(',')[1].split('.')[0])
        
        in_df = df.dropna(subset=["Embarked"])
        if not test:
            out_y = in_df[self.out_features]
        in_df = in_df[self.in_features]
        in_df.loc[in_df["Age"].isnull(), "Age"] = in_df["Age"].mean()
        in_df['Male'] = in_df['Sex'].map(lambda x: True if x=="male" else False)
        in_df['Title'] = self.extract_title(df['Name'])    

        titles = set(titles_map.values())
        for title in titles:
            in_df['is_' + title] = in_df['Title'].map(lambda x: True if x==title else False)

        for embarked in ['C', 'Q', 'S']:
            in_df['Embarked_' + embarked] = in_df['Embarked'].map(lambda x: True if x==embarked else False)

        in_df['FamilySize'] = in_df['SibSp'] + in_df['Parch']

        in_df.drop(columns="Title", inplace=True)
        in_df.drop(columns="Embarked", inplace=True)
        in_df.drop(columns="Sex", inplace=True)
        in_df.drop(columns="Parch", inplace=True)
        in_df.drop(columns="SibSp", inplace=True)

        if self.col_index_set:
            in_df = in_df.reindex(columns = self.train_dumm_cols, fill_value = False)
        else:
            in_df = pd.get_dummies(in_df)
            self.train_dumm_cols = in_df.columns
            self.col_index_set = True

        if test:
            return in_df
        
        return in_df, out_y



def preprocess_dataset(df, test=False):
    in_features = ['Pclass', 'Sex', 'SibSp', 'Parch']
    if test:
        return pd.get_dummies(df[in_features]) 
    
    out_features = ['Survived']
    return pd.get_dummies(df[in_features]), df[out_features]

def kDataSplit(k, i, data):
    val_ratio = 1.0 / k
    interval = len(data) * val_ratio
    interval = np.floor(interval).astype(np.int16)

    splits = []
    pool = np.array(range(len(data)))
    for j in range(int(1/val_ratio)):
        split = np.random.choice(pool, size=interval, replace=False)
        split = split.tolist()
        splits.append(split)
        pool = pool[np.isin(pool, split, invert=True)]
    #for j in range(len(splits)):
    #    print(len(splits[j]))

    #K-fold Cross-Validation    
    val_pool = splits[i]
    train_pool = []
    for j in range(len(splits)):
        if i == j:
            continue
        train_pool.append(splits[j])
    train_pool = np.hstack(train_pool).tolist()

    train_data = data.iloc(axis=0)[train_pool]
    val_data = data.iloc(axis=0)[val_pool]
    return train_data, val_data

def kCrossVal(k, model_class, model_params, data):
    model = model_class(**model_params)

    sum_score = 0.0

    for i in range(k):
        train_data, val_data = kDataSplit(k, i, data)
        
        X_train, y_train = preprocess_dataset(train_data)
        X_val, y_val = preprocess_dataset(val_data)

        model.fit(X_train, np.ravel(y_train))
        val_pred = model.predict(X_val)
        acc = accuracy_score(y_true=y_val, y_pred=val_pred)
        sum_score += acc
    return sum_score / k


def tune_model(df, n_trials, mlflow_exp_name):
    mlflow.set_experiment(mlflow_exp_name)
    experiment = mlflow.get_experiment_by_name(mlflow_exp_name)

    def objective(trial):  
        run = client.create_run(experiment.experiment_id)

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "random_state": 1,
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 7),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
        }
        for key in params:
            client.log_param(run.info.run_id, key, params[key])

        acc = kCrossVal(k=5,
                        model_class=RandomForestClassifier,
                        model_params=params,
                        data=df)

        client.log_metric(run.info.run_id, "accuracy", acc)
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_trial.params