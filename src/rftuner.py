"""Optuna tuner for RandomForest"""
import optuna
import mlflow
from optuna.integration.mlflow import MLflowCallback
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame

from utils import k_cross_val, DataPreprocessor


def tune_model(data : DataFrame,
               n_trials: int,
               study_name : str,
               preprocessor : DataPreprocessor,
               mlflow_tracking_uri : str) -> dict[str, any]:
    """RF paramater hyperoptimization

    Args:
        data (DataFrame): data, usually train dataset
        n_trials (int): number of trials to do
        study_name (str): MLFlow study name
        preprocessor (DataPreprocessor): data processing object
        mlflow_tracking_uri (str): MLFlow tracking uri

    Returns:
        dict[str, any]: Dictionary containing best model parameters 
    """

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mlflc = MLflowCallback(
        tracking_uri=mlflow_tracking_uri,
        metric_name="accuracy",
        create_experiment=True
    )

    #client = mlflow.tracking.MlflowClient()
    #mlflow.set_experiment(mlflow_exp_name)
    #experiment = mlflow.get_experiment_by_name(mlflow_exp_name)
    #print(str(experiment))

    @mlflc.track_in_mlflow()
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 450, 950, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "random_state": 1,
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 35),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 25),
            "max_features": trial.suggest_float("max_features", 0.2, 0.7)
        }

        mlflow.log_params(params)

        acc = k_cross_val(k=5,
                        model_class=RandomForestClassifier,
                        model_params=params,
                        data=data,
                        preprocessor=preprocessor)

        mlflow.log_metric("accuracy", acc)
        return acc

    study = optuna.create_study(study_name=study_name,
                                pruner=optuna.pruners.HyperbandPruner(),
                                direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, callbacks=[mlflc])
    return study.best_trial.params
