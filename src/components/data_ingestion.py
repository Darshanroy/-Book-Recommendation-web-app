import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    Books_data_path = os.path.join('artifacts','Books.csv')
    User_data_path = os.path.join('artifacts','Users.csv')
    Ratings_data_path = os.path.join('artifacts','Ratings.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Started')

        try:
            """
                Here we are reading the data from the local files and saving them in the folders,
                        this is because it looks like pulling the data from the database 
            """

            books_df = pd.read_csv('../pipelines/DATA/Books_dataset/Books.csv', on_bad_lines='skip')
            logging.info("Books Dataread Sucessfully")
            os.makedirs(os.path.dirname(self.ingestion_config.Books_data_path),exist_ok=True)
            books_df.to_csv(self.ingestion_config.Books_data_path,index=False,header=True)


            user_df = pd.read_csv('../pipelines/DATA/Users_dataset/Users.csv', on_bad_lines='skip')
            logging.info('User data read sucessfully')
            os.makedirs(os.path.dirname(self.ingestion_config.User_data_path), exist_ok=True)
            user_df.to_csv(self.ingestion_config.User_data_path, index=False, header=True)


            ratings_df =  pd.read_csv('../pipelines/DATA/Ratings_dataset/Ratings.csv', on_bad_lines='skip')
            logging.info('User data read sucessfully')
            os.makedirs(os.path.dirname(self.ingestion_config.Ratings_data_path), exist_ok=True)
            ratings_df.to_csv(self.ingestion_config.Ratings_data_path, index=False, header=True)


            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.Books_data_path,
                self.ingestion_config.User_data_path,
                self.ingestion_config.Ratings_data_path
            )

        except Exception as e:
            logging.info('Error occured in Data ingestion')
            raise CustomException(sys,e)

