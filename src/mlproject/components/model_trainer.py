import os
import pandas as pd
from src.mlproject.logging import logger
from sklearn.tree import DecisionTreeRegressor
import joblib
from src.mlproject.entity.config_entity import ModelTrainingConfig



class ModelTrainer:
    def __init__(self,config: ModelTrainingConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        
        train_X = train_data.drop([self.config.target_column],axis=1)
        test_X = test_data.drop([self.config.target_column],axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]
        


        DT = DecisionTreeRegressor(max_depth=self.config.max_depth,
                                   min_samples_split=self.config.min_samples_split,
                                   min_samples_leaf=self.config.min_samples_leaf,
                                   random_state=42)
        DT.fit(train_X,train_y)
        # y_pred =DT.predict(test_X)

        joblib.dump(DT, os.path.join(self.config.root_dir,self.config.model_name))