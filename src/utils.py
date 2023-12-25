import os
import sys
from src.exception import CustomException
import pickle


def save_object(file_path,object):
    try:
        dir_name=os.path.dirname(file_path) 
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,"wb") as f:
            pickle.dump(object,f)

    except Exception as e:
        raise CustomException(e,sys)