from src.mlproject.config.configuration import ConfigurationManager
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.logging import logger


STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config= ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            train,test = data_transformation.train_test_splitting()
            data_transformation.initiate_data_transformation(train_data=train,test_data=test)

        except Exception as e:
            raise e
    
        


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<\n\nx=================x")
    except Exception as e:
        raise e