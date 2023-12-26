import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

from src.logger import logging
class PredictPipeline:

    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            logging.info("Entered in prediction")
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','prepocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            prediction=model.predict(data_scaled)
            logging.info("Prediction for user data is done!")
            return prediction
            
        except Exception as e:
            raise CustomException(e,sys)
        
class UserData:
    def __init__(self,ph:int,
                Hardness:int,
                Solids:int,
                Chloramines:int,
                Sulfate:int,
                Conductivity:int,
                Organic_carbon:int,
                Trihalomethanes:int,
                Turbidity:int) :
        self.ph=ph
        self.Hardness=Hardness
        self.Solids=Solids
        self.Chloramines=Chloramines
        self.Sulfate=Sulfate
        self.Conductivity=Conductivity
        self.Organic_carbon=Organic_carbon
        self.Trihalomethanes=Trihalomethanes
        self.Turbidity=Turbidity

    def get_user_data_as_dataframe(self):
        try:
            logging.info("User data received")
            user_data={
            "ph":[self.ph],
            "Hardness":[self.Hardness],
            "Solids":[self.Solids],
            "Chloramines":[self.Chloramines],
            "Sulfate":[self.Sulfate],
            "Conductivity":[self.Conductivity],
            "Organic_carbon":[self.Organic_carbon],
            "Trihalomethanes":[self.Trihalomethanes],
            "Turbidity":[self.Turbidity]
            }
            logging.info("user data is converted into dataframe fro prediction")
            return pd.DataFrame(user_data)

        except Exception as e:
            raise CustomException(e,sys)