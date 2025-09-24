import joblib
import os
from app.utils.data_loader import DataLoader
from app.utils.data_cleaner import DataCleaner
from app.exception_logging.logger import logging
from app.utils.feature_engineering import FeatureEngineer
from app.utils.feature_engineering import FeatureEngineer
from app.utils.model_trainer import ModelTrainer
from sklearn.preprocessing import LabelEncoder
from app.config import RANDOM_STATE, TEST_SIZE, CROSS_VAL_FOLDS, RANDOM_SEARCH_PARAMS, FILE_PATH

def main():

    data_loader = DataLoader(file_path=FILE_PATH)

    data_cleaner = DataCleaner(data_loader=data_loader)
    X, y = data_cleaner.clean_data()

    preprocessor = FeatureEngineer(data_cleaner=data_cleaner).engineer_features()

    model_tuner = ModelTrainer(X=X, y=y)

    X_train, X_test, y_train, y_test, model = model_tuner.train_model()

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)


    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    train_model = model.fit(X_train_transformed, y_train_encoded)

    tuned_model = train_model.best_estimator_

    classification_metrics = model_tuner.get_classification_score(y_test_encoded, tuned_model.predict(X_test_transformed))

    try:
        os.makedirs('preprocessor', exist_ok=True)
        joblib.dump(preprocessor, "preprocessor/preprocessor.pkl")
    except:
        logging.error("Error saving the preprocessor locally.")

    try:
        os.makedirs('models', exist_ok=True)
        joblib.dump(tuned_model, "models/breast_cancer_prediction_model.pkl")
    except:
        logging.error("Error saving the model locally.")

    model_tuner.track_mlflow_experiment(tuned_model, classification_metrics)

    








if __name__ == "__main__":
    main()