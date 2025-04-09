import os
from src.mlproject.logging import logger
from src.mlproject.entity.config_entity import DataValidationConfig
import pandas as pd



class DataValidation:
    def __init__(self,config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()
           
           
            for col in all_cols:
                if col not in list[all_schema]:
                    validation_status = False
                    with open(self.config.STATUS_FILE,'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    logger.info("status is True ")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE,'w') as f:
                        f.write(f"Validation Status: {validation_status}")
                    logger.info("status is False ")

            return validation_status
        
        except Exception as e:
            raise e




