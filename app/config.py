TEST_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VAL_FOLDS = 5

FILE_PATH = 'data/Breast_cancer_dataset.csv'
MODEL_PATH = 'models/breast_cancer_model.pkl'


RANDOM_SEARCH_PARAMS = {
    'XGBClassifier': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    },
    'DecisionTree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
}