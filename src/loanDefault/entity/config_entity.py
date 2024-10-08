from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    x_train_data_path: Path
    x_test_data_path: Path
    y_train_data_path: Path
    y_test_data_path: Path
    model_name: str
    learning_rate: float
    n_estimators: int
    max_depth: int
    min_child_weight: int
    gamma: int
    subsample: float
    colsample_bytree: float
    objective: str
    nthread: int
    scale_pos_weight: int
    seed: int
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    x_test_data_path: Path
    y_test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str