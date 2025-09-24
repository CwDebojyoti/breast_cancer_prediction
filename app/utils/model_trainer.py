from app.exception_logging.logger import logging
from app.utils.data_loader import DataLoader
from app.utils.data_cleaner import DataCleaner
from app.utils.feature_engineering import FeatureEngineer
from app.config import RANDOM_STATE, TEST_SIZE, CROSS_VAL_FOLDS, RANDOM_SEARCH_PARAMS

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score,precision_score,recall_score

import mlflow
import mlflow.sklearn

from app.entity.artifact_entity import ClassificationMetricArtifact


class ModelTrainer:
    def __init__(self, X, y, feature_engineer: FeatureEngineer = None):
        self.X = X
        self.y = y

    def get_classification_score(self, y_true, y_pred)->ClassificationMetricArtifact:
        try:   
            model_f1_score = f1_score(y_true, y_pred)
            model_recall_score = recall_score(y_true, y_pred)
            model_precision_score=precision_score(y_true,y_pred)

            classification_metric =  ClassificationMetricArtifact(f1_score=model_f1_score,
                        precision_score=model_precision_score, 
                        recall_score=model_recall_score)
            return classification_metric
        except Exception as e:
            logging.error(f"Error in get_classification_score: {e}")
            raise
        

    def train_model(self):
        try:
            logging.info("Starting model training process.")
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            logging.info("Data split into training and testing sets.")

            model = XGBClassifier()
            # model = DecisionTreeRegressor()

            
            logging.info("Initialized DecisionTree model trainer with RandomSearch tuner.")
            randomsearch_model_tune = RandomizedSearchCV(estimator=model,
                                                          param_distributions=RANDOM_SEARCH_PARAMS['XGBClassifier'],
                                                          n_iter=100,
                                                          cv=CROSS_VAL_FOLDS,
                                                          scoring='neg_mean_squared_error',
                                                          verbose=2,
                                                          random_state=RANDOM_STATE,
                                                          n_jobs=-1)
            
            logging.info("Configured RandomizedSearchCV for hyperparameter tuning.")

            logging.info("Model trainer setup complete. Ready to fit the model.")
            

            return X_train, X_test, y_train, y_test, randomsearch_model_tune

        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise

    def track_mlflow_experiment(self, best_model, classificationmetric):
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score

            

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")