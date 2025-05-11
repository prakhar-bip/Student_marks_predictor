import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.components.utils import *


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("datasets", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test input data")

            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Desicion tree": DecisionTreeRegressor(),
                "XGBoost": XGBRegressor(),
                "gradient boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "Linear regression": LinearRegression(),
            }

            params = {
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                },
                "Desicion tree": {
                    "max_depth": [10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                },
                "XGBoost": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                },
                "gradient boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "subsample": [0.6, 0.8, 1.0],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"],
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["ball_tree", "kd_tree", "brute"],
                    "p": [1, 2],
                },
                "Linear regression": {},
            }

            model_report: dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model is found")

            logging.info("Best model found")

            save_obj(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model,
            )

            predictions = best_model.predict(X_test)

            return (r2_score(y_test, predictions), best_model_name)
        except Exception as e:
            raise CustomException(e, sys)
