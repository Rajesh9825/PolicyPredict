import os
from src.mlproject.logging import logger
from src.mlproject.utils.common import get_size
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from src.mlproject.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion: 
    def __init__(self,config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_file_path):
            # Load environment variables from .env file
            load_dotenv()

            # Authenticate with Kaggle
            api = KaggleApi()
            api.authenticate()

            # Download dataset
            api.dataset_download_files(self.config.source_URL, path=self.config.unzip_dir, unzip=True)
            logger.info(f"{self.config.source_URL} dataset downloaded! On following path: \n{self.config.unzip_dir}")

        else:
            logger.info(f"file already exists of size: {get_size(Path(self.config.local_file_path))}")
        


