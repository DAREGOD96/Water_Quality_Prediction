import os 
import sys


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model,save_object

@dataclass
class ModelTrainerConfig:
    best_model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config_object=ModelTrainerConfig()

    def model_trainer_initiated(self,train_arr,test_arr):
        try:

            logging.info("Initiated Model training")
            X_train,y_train,X_test,y_test= (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("splitting the dataset as X_train,y_train,X_test,y_test is done")

            models={
                "RandomForestClassifier":RandomForestClassifier(),
                "SVC":SVC(),
                "LogisticRegression":LogisticRegression(),
                "KNeighborsClassifier":KNeighborsClassifier(),
                "DecisionTreeClassifier":DecisionTreeClassifier()
            }
            params = {
                "RandomForestClassifier": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "SVC": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                },
                "LogisticRegression": {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2']
                },
                "KNeighborsClassifier": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                "DecisionTreeClassifier": {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            best_model_score=max(sorted(model_report.values()))
            best_model = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            logging.info(f"Best performing model is {best_model} with score of {best_model_score}")

            if best_model_score<0.6:
                raise CustomException("No best model found")
            save_object(
                file_path=self.model_trainer_config_object.best_model_path,
                object=best_model
            )
            return best_model_score
        except Exception as e:
            raise CustomException(e,sys)