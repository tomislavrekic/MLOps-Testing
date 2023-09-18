import optuna
import mlflow
from sklearn.ensemble import RandomForestClassifier
from utils import kCrossVal

def tune_model(df, n_trials, mlflow_exp_name, preprocessor):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    client = mlflow.tracking.MlflowClient()
    mlflow.set_experiment(mlflow_exp_name)
    experiment = mlflow.get_experiment_by_name(mlflow_exp_name)
    print(str(experiment))

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
                        data=df, 
                        preprocessor=preprocessor)

        client.log_metric(run.info.run_id, "accuracy", acc)
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_trial.params