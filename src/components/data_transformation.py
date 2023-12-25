import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    prepocessor_obj_file_path= os.path.join("artifacts","prepocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns=[
                "ph",
                "Hardness",
                "Solids",
                "Chloramines",
                "Sulfate",
                "Conductivity",
                "Organic_carbon",
                "Trihalomethanes",
                "Turbidity"
            ]

            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",MinMaxScaler())
                ]
            )

            prepocessor= ColumnTransformer([("num_pipeline",numerical_pipeline, numerical_columns)])        
            return prepocessor
        
        except Exception as e :
            raise CustomException(e,sys)
    
    def data_transformation_initiated(self,train_set,test_set):
        try:

            train_df=pd.read_csv(train_set)
            test_df=pd.read_csv(test_set)
            logging.info("train and test set received for prepocessing")

            prepocessor_obj= self.get_data_transformer_object()

            target_column_name="Potability"

            X_train=train_df.drop(columns=[target_column_name],axis=1)
            y_train=train_df[target_column_name]

            X_test=test_df.drop(columns=[target_column_name],axis=1)
            y_test=test_df[target_column_name]

            logging.info("train and test data is further splitted into X_train,y_train,X_test,y_test")

            X_train_arr = prepocessor_obj.fit_transform(X_train)
            X_test_arr = prepocessor_obj.transform(X_test)

            logging.info("prepocessing of the numerical columns is done!")

            train_arr=np.c_[X_train_arr,np.array(y_train)]
            test_arr=np.c_[X_test_arr,np.array(y_test)]

            logging.info("X_train,y_train,X_test,y_test are added together")

            save_object(
                file_path=self.data_transformation_config.prepocessor_obj_file_path,
                object=prepocessor_obj
            )

            logging.info("Data transformation is done!")


            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            raise CustomException(e,sys)

