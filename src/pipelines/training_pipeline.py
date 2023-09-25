import os
import sys

import numpy as np

from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.components.model_training import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import save_object

if __name__=='__main__':
    ingestion_obj=DataIngestion()
    book,user,ratings=ingestion_obj.initiate_data_ingestion()

    transformation_obj = DataTransformation()
    preprocesed_file = transformation_obj.initiate_data_tranformation_object(book,user,ratings)

    modeltrainer_obj = ModelTrainer()
    model_knn, us_canada_user_rating_pivot = modeltrainer_obj.initate_model_training(preprocesed_file)


    save_object(
        file_path=os.path.join('artifacts','us_canada_user_rating_pivot.pkl'),
        obj=us_canada_user_rating_pivot
    )
    save_object(
        file_path=os.path.join('artifacts', 'model_knn.pkl'),
        obj=model_knn
    )








