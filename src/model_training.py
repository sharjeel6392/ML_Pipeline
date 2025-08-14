import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# Ensure the "logs" directory exists
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger('model_training')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.
    
    Parameters:
    - params_path: str, path to the YAML file
    
    Returns:
    - Dictionary containing the parameters
    """
    try:
        logger.info(f"Loading parameters from {params_path}")
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info("Parameters loaded successfully")
        return params
    except FileNotFoundError as e:
        logger.error(f"Parameters file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading parameters: {e}")
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Parameters:
    - data_url: str, path to the CSV file
    
    Returns:
    - DataFrame containing the loaded data
    """
    try:
        logger.info(f"Loading data from {data_url}")
        data = pd.read_csv(data_url)
        logger.info("Data loaded successfully")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise   
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading data: {e}")
        raise

def train_model(train_data: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train a Random Forest model.
    
    Parameters:
    - train_data: np.ndarray, training data features
    - y_train: np.ndarray, training data labels
    - params: dict, parameters for the Random Forest model
    
    Returns:
    - RandomForestClassifier, trained model
    """
    try:
        if train_data.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in train_data and y_train must match.")
        logger.info("Starting model training")
        logger.info("Training model with parameters: {}".format(params))
        model = RandomForestClassifier(n_estimators=params.get('n_estimators', 100),
                                       max_depth=params.get('max_depth', None),
                                       random_state=params.get('random_state', 42))
        model.fit(train_data, y_train)
        logger.info("Model trained successfully")
        return model
    except ValueError as e:
        logger.error(f"Value error during model training: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """
    Save the trained model to a file.
    
    Parameters:
    - model: RandomForestClassifier, trained model
    - model_path: str, path to save the model
    """
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully at {model_path}")
    except FileNotFoundError as e:
        logger.error(f"File not found while saving model: {e}")
        raise
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    try:
        params = load_params('params.yaml')
        param = {
            'n_estimators': params.get('n_estimators', 100),
            'max_depth': params.get('max_depth', None),
            'random_state': params.get('random_state', 42)
        }
        # Load preprocessed data
        train_data = pd.read_csv('./data/processed/train_tfidf.csv')
        test_data = pd.read_csv('./data/processed/test_tfidf.csv')
        x_train = train_data.drop(columns=['target']).values
        y_train = train_data['target'].values
        x_test = test_data.drop(columns=['target']).values

        clf = train_model(x_train, y_train, param)
        model_path = './models/trained_model.pkl'
        save_model(clf, model_path)
        logger.info("Model training and saving completed successfully")

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise

if __name__ == "__main__":
    main()