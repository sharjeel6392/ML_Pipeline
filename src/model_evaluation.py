import os
import pandas as pd
import numpy as np
import pickle
import logging
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Ensure the "logs" directory exists
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(model_path: str) -> object:
    """
    Load a trained model from a file.
    
    Parameters:
    - model_path: str, path to the model file
    
    Returns:
    - Loaded model object
    """
    try:
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info("Model loaded successfully")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading model: {e}")
        raise

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Parameters:
    - data_path: str, path to the CSV file
    
    Returns:
    - DataFrame containing the loaded data
    """
    try:
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        logger.info("Data loaded successfully")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading data: {e}")
        raise

def evaluate_model(model: object, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate the model using various metrics.
    
    Parameters:
    - model: object, trained model
    - X: np.ndarray, features for evaluation
    - y: np.ndarray, true labels for evaluation
    
    Returns:
    - dict, evaluation metrics
    """
    try:
        logger.info("Evaluating model")
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted')
        recall = recall_score(y, predictions, average='weighted')
        roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_evaluation_results(metrics: dict, output_path: str) -> None:
    """
    Save the evaluation metrics to a JSON file.
    
    Parameters:
    - metrics: dict, evaluation metrics
    - output_path: str, path to save the JSON file
    """
    try:
        logger.info(f"Saving evaluation results to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info("Evaluation results saved successfully")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        raise

def main():
    """
    Main function to load model, data, evaluate the model, and save results.
    """
    try:
        # Load model
        model_path = './models/trained_model.pkl'
        model = load_model(model_path)
        
        # Load data
        data_path = './data/processed/test_tfidf.csv'
        data = load_data(data_path)
        
        # Prepare features and labels
        X = data.drop(columns=['target']).values
        y = data['target'].values
        
        # Evaluate model
        metrics = evaluate_model(model, X, y)
        
        # Save evaluation results
        output_path = './reports/evaluation_report.json'
        save_evaluation_results(metrics, output_path)
        
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise

if __name__ == "__main__":
    main()