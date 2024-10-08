import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from src.loanDefault.entity.config_entity import ModelEvaluationConfig
from src.loanDefault.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        # Compute confusion matrix and classification report
        conf_matrix = confusion_matrix(actual, pred)
        class_report = classification_report(actual, pred, output_dict=True)

        return conf_matrix, class_report

    def log_into_mlflow(self):
        test_x = pd.read_csv(self.config.x_test_data_path)
        test_y = pd.read_csv(self.config.y_test_data_path)

        model = joblib.load(self.config.model_path)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_classes = model.predict(test_x)

            # Evaluate metrics for classification
            conf_matrix, class_report = self.eval_metrics(test_y, predicted_classes)

            # Save classification report as JSON using save_json
            metrics_file_path = Path(self.config.metric_file_name)
            save_json(path=metrics_file_path, data=class_report)

            # Log the classification report metrics into mlflow
            for label, metrics in class_report.items():
                if isinstance(metrics, dict):  # Skip the 'accuracy' entry which is not a dict
                    mlflow.log_metric(f'{label}_precision', metrics['precision'])
                    mlflow.log_metric(f'{label}_recall', metrics['recall'])
                    mlflow.log_metric(f'{label}_f1-score', metrics['f1-score'])
                    mlflow.log_metric(f'{label}_support', metrics['support'])

            # Log the overall accuracy
            mlflow.log_metric('accuracy', class_report['accuracy'])

            mlflow.log_params(self.config.all_params)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="XGBModel")
            else:
                mlflow.sklearn.log_model(model, "model")