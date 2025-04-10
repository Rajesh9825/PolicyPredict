import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.preprocessor = joblib.load(Path('artifacts/data_transformation/preprocessor.joblib'))

    def transform(self,data):
        data = pd.DataFrame(data,columns=['age','sex','bmi','children','smoker','region'])
        print(data)
        preprocessor = self.preprocessor.transform(data)
        
        return preprocessor
      
    def predict(self,data):
        
        prediction = self.model.predict(data)

        return prediction