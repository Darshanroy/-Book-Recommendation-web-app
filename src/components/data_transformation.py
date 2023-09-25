from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from scipy.sparse import csr_matrix



@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')



class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTranformationConfig()


    def initiate_data_tranformation_object(self,books_path,users_path,ratings_path):
        try:
            logging.info("Data tranformation initated")

            books_df = pd.read_csv(books_path, on_bad_lines='skip')
            user_df = pd.read_csv(users_path, on_bad_lines='skip')
            ratings_df = pd.read_csv(ratings_path, on_bad_lines='skip')

            #Lowering the Column Names
            books_df.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS',
                                'imageUrlM', 'imageUrlL']
            user_df.columns = ['userID', 'Location', 'Age']
            ratings_df.columns = ['userID', 'ISBN', 'bookRating']


            #refer to documentaion for explaination
            counts1 = ratings_df['userID'].value_counts()
            ratings_df = ratings_df[ratings_df['userID'].isin(counts1[counts1 >= 200].index)]

            counts = ratings_df['bookRating'].value_counts()
            ratings_df = ratings_df[ratings_df['bookRating'].isin(counts[counts >= 100].index)]

            #Combaing the data_frames using the same ISBN number
            combine_book_rating = pd.merge(ratings_df, books_df, on='ISBN')
            columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
            combine_book_rating = combine_book_rating.drop(columns, axis=1)

            combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])


            #refer to documentaion
            book_ratingCount = (combine_book_rating.
            groupby(by=['bookTitle'])['bookRating'].
            count().
            reset_index().
            rename(columns={'bookRating': 'totalRatingCount'})
            [['bookTitle', 'totalRatingCount']]
            )

            rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on='bookTitle',
                                                                     right_on='bookTitle', how='left')
            pd.set_option('display.float_format', lambda x: '%.3f' % x)

            popularity_threshold = 50
            rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

            combined = rating_popular_book.merge(user_df, left_on='userID', right_on='userID', how='left')

            us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada|india")]
            us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)

            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=us_canada_user_rating

            )
            logging.info("Data Tranformation Complted")

            #return the dataframe
            return (
                us_canada_user_rating
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e, sys)






