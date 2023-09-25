from urllib import request
from flask import Flask, render_template,request
import os
import sys

import numpy as np
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging
import pickle
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



def get_recommendations():
    try:
        model_knn = load_object('src/pipelines/artifacts/model_knn.pkl')
        us_canada_user_rating_pivot = load_object('src/pipelines/artifacts/us_canada_user_rating_pivot.pkl')
        logging.info("retriveing_list_of_recomendations")
        list_of_books=[]
        query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
        us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1)
        distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1),n_neighbors=6)

        for i in range(0, len(distances.flatten())):
            if i == 0:

                title=us_canada_user_rating_pivot.index[query_index]
            else:
                # print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]],
                #                                                distances.flatten()[i]))

                list_of_books.append((us_canada_user_rating_pivot.index[indices.flatten()[i]],distances.flatten()[i]))

        return list_of_books,title

    except Exception as e:
        raise CustomException(e,sys)
@app.route('/recommend', methods=['POST','GET'])
def recommend():
    # Check if the user clicked "Yes"
    if request.method == 'POST' and 'yes_button' in request.form:
        recommendations,title = get_recommendations()
        return render_template('recommendations.html', recommendations=recommendations,title=title)
    # Check if the user clicked "No"
    elif request.method == 'POST' and 'no_button' in request.form:
        return render_template('goodbye.html')  # Redirect to a "goodbye" page or handle accordingly

    else:
        # Handle other cases or errors
        return "Invalid request"
if __name__ == '__main__':
    app.run(debug=True)

