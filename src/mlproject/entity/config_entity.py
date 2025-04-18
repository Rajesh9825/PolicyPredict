## Configuration entity
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_file_path: Path
    unzip_dir: Path
    

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir : Path
    STATUS_FILE : str
    unzip_data_dir : Path
    all_schema : dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir : Path
    data_path : Path
    

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    max_depth: int # params
    min_samples_split: int
    min_samples_leaf: int
    target_column: str # Schema.yaml



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str