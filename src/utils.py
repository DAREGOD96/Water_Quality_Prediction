import os
import sys
from src.exception import CustomException
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from src.logger import logging
import warnings
def save_object(file_path,object):
    try:
        dir_name=os.path.dirname(file_path) 
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,"wb") as f:
            pickle.dump(object,f)

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:

        logging.info("Entered in model training")
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        logging.info("Model training is completed!")
        return report
    except Exception as e:
        raise CustomException(e,sys)