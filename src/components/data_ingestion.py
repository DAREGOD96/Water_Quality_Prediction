import os 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts","raw_data.csv")
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config_object=DataIngestionConfig()

    def data_ingestion_initiated(self):
        try:
            df=pd.read_csv("dataset\water_potability.csv")
            logging.info("Reading the dataset as dataframe done!")

            os.makedirs(os.path.dirname(self.data_ingestion_config_object.train_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config_object.raw_data_path,index=False,header=True)
            logging.info("saving the raw data in artifacts folder is done")

            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)
            logging.info("splitting the data into train set and test set is done")

            train_set.to_csv(self.data_ingestion_config_object.train_data_path,index=False,header=True)
            logging.info("saving the train data in the artifacts folder is done")

            test_set.to_csv(self.data_ingestion_config_object.test_data_path,index=False,header=True)
            logging.info("saving the test data in the artifacts folder is done")

            logging.info("Data ingestion is completed!")

            return(
                self.data_ingestion_config_object.train_data_path,
                self.data_ingestion_config_object.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    data_ingestion_object=DataIngestion()
    train_set,test_set=data_ingestion_object.data_ingestion_initiated()

    data_transformation_object=DataTransformation()
    train_arr,test_arr=data_transformation_object.data_transformation_initiated(train_set,test_set)

