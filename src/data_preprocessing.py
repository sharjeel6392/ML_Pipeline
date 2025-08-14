import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the "logs" directory exists
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text: str) -> str:
    """
    Transform text by removing punctuation, converting to lowercase, and stemming.
    
    Parameters:
    - text: str, input text
    
    Returns:
    - str, transformed text
    """
    try:
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        # Stem words
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        transformed_text = ' '.join(tokens)
        return transformed_text
    except Exception as e:
        logger.error(f"Error transforming text: {e}")
        raise

def preprocess_data(data: pd.DataFrame, text_column = 'text', target_column = 'target') -> pd.DataFrame:
    """
    Preprocess the data by handling missing values, removing duplicates and encoding categorical variables.
    
    Parameters:
    - data: DataFrame, raw data
    
    Returns:
    - DataFrame, preprocessed data
    """
    try:
        logger.info("Starting data preprocessing")
        
        # Encode the target column
        encoder = LabelEncoder()
        data[target_column] = encoder.fit_transform(data[target_column])
        logger.info("Target column encoded successfully")

        # Remove duplicates
        data.drop_duplicates(inplace=True, keep= 'first')
        logger.info("Duplicates removed successfully")

        # Apply text transformation on the specified text column
        data[text_column] = data[text_column].apply(transform_text)
        logger.info("Text transformation applied successfully")
        return data
    
    except KeyError as e:
        logger.error(f"Missing column in the dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise

def main(text_column = 'text', target_column = 'target') -> None:
    """
    Main function to load, preprocess, and save the data.
    
    Parameters:
    - data_url: str, path to the CSV file
    - output_path: str, path to save the preprocessed data
    """
    try:
        # Load data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
    
        logger.info("Data loaded successfully")
        
        # Preprocess data
        train_processed_data = preprocess_data(train_data, text_column=text_column, target_column=target_column)
        test_processed_data = preprocess_data(test_data, text_column=text_column, target_column=target_column)
        
        # Save preprocessed data
        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logger.info("Preprocessed data saved successfully")
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise


if __name__ == "__main__":
    main()