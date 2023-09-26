import optuna
import mlflow
from sklearn.ensemble import RandomForestClassifier
from utils import kCrossVal
from optuna.integration.mlflow import MLflowCallback

mlflc = MLflowCallback(
    tracking_uri="sqlite:///../mlflow.db",
    metric_name="accuracy",
    create_experiment=True
)


def tune_model(df, n_trials, study_name, preprocessor, mlflow_tracking_uri):
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

        acc = kCrossVal(k=5,
                        model_class=RandomForestClassifier,
                        model_params=params,
                        data=df, 
                        preprocessor=preprocessor)

        mlflow.log_metric("accuracy", acc)
        return acc

    study = optuna.create_study(study_name=study_name,
                                pruner=optuna.pruners.HyperbandPruner(),
                                direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, callbacks=[mlflc])
    return study.best_trial.params