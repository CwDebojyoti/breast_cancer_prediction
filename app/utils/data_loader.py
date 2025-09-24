import pandas as pd
import numpy as np
from app.exception_logging.logger import logging


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully with shape {data.shape}")

            try:
                data.drop(columns=['Unnamed: 32', 'id'], inplace=True)
            except KeyError as e:
                logging.warning(f"Columns to drop not found: {e}")

            try:
                X = data.drop(columns=['diagnosis'], axis=1)
                y = data['diagnosis']
            except KeyError as e:
                logging.error(f"Target column 'diagnosis' not found: {e}")
                raise
                
            return X, y

        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise