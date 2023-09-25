import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from src.utils import save_object

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, us_canada_user_rating):
        try:
            logging.info("model training process started")
            us_canada_user_rating_pivot = us_canada_user_rating.groupby(['bookTitle', 'userID'])[
                'bookRating'].mean().unstack().fillna(0)

            us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

            model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
            model_knn.fit(us_canada_user_rating_matrix)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model_knn
            )

            return model_knn, us_canada_user_rating_pivot

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e, sys)
















