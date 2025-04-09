
import os
import numpy as np
import pandas as pd
from src.mlproject.logging import logger
from src.mlproject.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import joblib



class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config = config


    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        # Split data into training and test set. (0.75, 0.25) split.
        train,test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir,"train.csv"),index= False)
        test.to_csv(os.path.join(self.config.root_dir,"test.csv"),index= False)

        logger.info("Splitted data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        return train,test

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''

        try:
            numerical_columns = ["age","bmi","children"]
            categorical_column = [
                "sex",
                "smoker",
                "region"
            ]

            # num_pipeline =Pipeline(
            #     steps=[
            #         ("imputer",SimpleImputer(strategy="median")),
            #         #("scaler",StandardScaler())
            #     ]
            # )

            cat_pipeline = Pipeline(
                steps=[
                    # ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    #("scaler",StandardScaler())
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    # ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_column),
                    
                ],remainder='passthrough'
            )

            logger.info("Preprocessor object created")

            return preprocessor
        

        except Exception as e:
            raise e
        
    
    def initiate_data_transformation(self,train_data,test_data):

        try:
            
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "charges"


            input_feature_train_df = train_data.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_data[target_column_name]
         
            input_feature_test_df = test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_data[target_column_name]

            logger.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]


            #### For dataframe output 
            transformed_train=pd.DataFrame(train_arr)
            transformed_test=pd.DataFrame(test_arr)

            
            transformed_train.to_csv(os.path.join(self.config.root_dir,"transformed_train.csv"),index= False)
            transformed_test.to_csv(os.path.join(self.config.root_dir,"transformed_test.csv"),index= False)

            logger.info(f"Saved train and test data.")

            joblib.dump(preprocessor_obj, os.path.join(self.config.root_dir,"preprocessor.joblib"))
            logger.info(f"Saved Preprocessing objects.")

            return train_arr,test_arr

        except Exception as e:
            raise e


