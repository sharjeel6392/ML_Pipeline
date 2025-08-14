import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

# Ensure the "logs" directory exists
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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
        logger.error(f"Unexpected error occured while loading data: {e}")
        raise

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and encoding categorical variables.
    
    Parameters:
    - data: DataFrame, raw data
    
    Returns:
    - DataFrame, preprocessed data
    """
    try:
        logger.info("Starting data preprocessing")
        data.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        data.rename(columns={
            'v1' : 'target',
            'v2' : 'text'
        }, inplace=True)
        logger.debug(f"Data preprocessing completed. Columns after renaming: {data.columns.tolist()}")
        return data
    except KeyError as e:
        logger.error(f"Missing column in the dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_path: str) -> None:
    """
    Save the preprocessed train and test data to their respective CSV file.
    
    Parameters:
    - train_data, test_data: DataFrame, preprocessed data
    - output_path: str, path to save the CSV file
    """
    try:
        raw_data_path = os.path.join(output_path, 'data.csv')
        os.makedirs(output_path, exist_ok=True)
        train_data.to_csv(os.path.join(output_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(output_path, 'test.csv'), index=False)
        logger.info(f"Saving train and test data to {output_path}")
        logger.debug("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")
        raise

def main():
    """
    Main function to execute the data ingestion pipeline.
    """
    try:
        test_size = 0.2
        data_url = './experiments/spam.csv'  # Update this path as needed
    
        # Load data
        data = load_data(data_url)
        
        # Preprocess data
        preprocessed_data = preprocess_data(data)
        
        # Split data into train and test sets
        train_data, test_data = train_test_split(preprocessed_data, test_size=test_size, random_state=42)

        # Save the processed data
        data_path = os.path.join('./data', 'raw')
        save_data(train_data, test_data, output_path=data_path)
        
        logger.info("Data ingestion pipeline completed successfully")
    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")

if __name__ == "__main__":
    main()