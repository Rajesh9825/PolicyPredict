import os
import pandas as pd
from src.mlproject.utils.common import save_json
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlflow.models.signature import infer_signature
from src.mlproject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path


# import dagshub
# dagshub.init(repo_owner='rajuu9825', repo_name='my-first-repo', mlflow=True)


class ModelEvaluation:
    def __init__(self,config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        
        return rmse,mae,r2
    
    def log_into_mlflow(self):

        # Load test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[[self.config.target_column]]

        # Set MLflow Tracking URI (DagsHub or local)
        # mlflow.set_tracking_uri(self.config.mlflow_uri)  # e.g., https://dagshub.com/your_username/your_repo.mlflow

        # Optional: set username/token as env vars if needed
        # os.environ["MLFLOW_TRACKING_USERNAME"] = "your_username"
        # os.environ["MLFLOW_TRACKING_PASSWORD"] = "your_token"

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Set or create experiment
        exp = mlflow.set_experiment(experiment_name="exp-1")

        with mlflow.start_run(experiment_id=exp.experiment_id):
            # Predictions and signature
            pred = model.predict(X_test)
            # signature = infer_signature(X_test, pred)

            # Evaluation
            rmse, mae, r2 = self.eval_metrics(y_test, pred)

            # Save metrics locally (optional)
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log params and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(model,"DecisionTree")

            # Log model and register it
            # if tracking_url_type_store != "file":
            #     mlflow.sklearn.log_model(
            #         sk_model=model,
            #         artifact_path="model",
            #         registered_model_name="PolicyPredict",  # This name will show in Model Registry
            #         signature=signature
            #     )
            # else:
            #     mlflow.sklearn.log_model(
            #         sk_model=model,
            #         artifact_path="model",
            #         signature=signature
            #     )

            # print("MLflow logging complete 🚀")
