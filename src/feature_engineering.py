import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

# Ensure the "logs" directory exists
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, 'feature_engineering.log')
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
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading data: {e}")
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply TF-IDF vectorization to the DataFrame.
    
    Parameters:
    - train_data: pd.DataFrame, input training data
    - test_data : pd.DataFrame, input test data
    - max_features: int, maximum number of features to extract
    
    Returns:
    - tuple of DataFrames, transformed train and test data
    """
    try:
        logger.info("Applying TF-IDF vectorization")
        vectorizer = TfidfVectorizer(max_features=max_features)
        x_train = train_data['text'].fillna('').values
        x_test = test_data['text'].fillna('').values
        y_train = train_data['target'].values
        y_test = test_data['target'].values
        train_tfidf = vectorizer.fit_transform(x_train)
        test_tfidf = vectorizer.transform(x_test)
        
        train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        train_tfidf_df['target'] = y_train
        test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        test_tfidf_df['target'] = y_test
        
        logger.info("TF-IDF vectorization applied successfully")
        return train_tfidf_df, test_tfidf_df
    except Exception as e:
        logger.error(f"Error applying TF-IDF vectorization: {e}")
        raise

def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the DataFrame to a CSV file.
    
    Parameters:
    - df: DataFrame, data to save
    - output_path: str, path to save the CSV file
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Saving data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    """
    Main function to load, apply TF-IDF, and save the data.
    """
    try:
        params = load_params('params.yaml')
        max_features = params.get('feature_engineering', {}).get('max_features', 100)
        logger.info(f"Max features for TF-IDF: {max_features}")
        
        # Load data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        
        # Apply TF-IDF
        train_tfidf_df, test_tfidf_df = apply_tfidf(train_data, test_data, max_features)
        
        # Save transformed data
        save_data(train_tfidf_df, './data/processed/train_tfidf.csv')
        save_data(test_tfidf_df, './data/processed/test_tfidf.csv')
        
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise

if __name__ == "__main__":
    main()