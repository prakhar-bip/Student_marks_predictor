import pandas as pd
import sys
import os
from src.exception import CustomException
from src.components.utils import *


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "datasets\\model.pkl"
            preprocessor_path = "datasets\\preprocessor.pkl"
            model = load_model(file_path=model_path)
            preprocessor = load_model(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethinicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: int,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethinicity = race_ethinicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethinicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

