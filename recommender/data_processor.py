# Data loading and preprocessing

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sys
from pathlib import Path

# Add parent directory to path for logger import
sys.path.append(str(Path(__file__).parent.parent))
from logger_config import setup_logger

# Setup logger
logger = setup_logger("data_processor")

# Download required NLTK data
logger.info("Checking for required NLTK data")
try:
    nltk.data.find('tokenizers/punkt')
    logger.debug("NLTK punkt tokenizer found")
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer")
    nltk.download('punkt', quiet=True)
    logger.info("NLTK punkt tokenizer downloaded")

try:
    nltk.data.find('corpora/stopwords')
    logger.debug("NLTK stopwords corpus found")
except LookupError:
    logger.info("Downloading NLTK stopwords corpus")
    nltk.download('stopwords', quiet=True)
    logger.info("NLTK stopwords corpus downloaded")

# Initialize stemmer
ps = PorterStemmer()

def load_data(file_path):
    """
    Load the processed CSV file.
    
    Args:
        file_path (str): Path to the processed CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}, Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        raise


def clean_text(text):
    """
    Clean text by removing special characters and converting to lowercase.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    # Convert to string and lowercase
    text = str(text).lower()
    # Remove special characters, keep only alphanumeric and spaces
    text = re.sub('[^a-zA-Z0-9\s]', ' ', text)
    # Remove extra whitespaces
    text = re.sub('\s+', ' ', text).strip()
    return text


def stem_text(text):
    """
    Apply stemming to text using Porter Stemmer.
    
    Args:
        text (str): Input text to stem
        
    Returns:
        str: Stemmed text
    """
    if not text:
        return ""
    
    # Tokenize the text
    words = text.split()
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Stem each word and remove stopwords
    stemmed_words = [ps.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(stemmed_words)


def organize_keywords(df):
    """
    Organize and combine keywords from product name and other relevant fields.
    Apply text cleaning and stemming.
    
    Args:
        df (pd.DataFrame): Input dataframe with product data
        
    Returns:
        pd.DataFrame: Dataframe with processed 'tags' column
    """
    logger.info(f"Organizing keywords for dataframe with shape {df.shape}")
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Combine name and other text fields for better keyword extraction
    # Primary focus on 'name' field as it contains most relevant keywords
    logger.debug("Combining text fields")
    df['combined_text'] = df['name'].fillna('')
    
    # Clean the text
    logger.debug("Cleaning text")
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Apply stemming
    logger.debug("Applying stemming")
    df['tags'] = df['cleaned_text'].apply(stem_text)
    
    # Drop intermediate columns
    df.drop(['combined_text', 'cleaned_text'], axis=1, inplace=True)
    
    # Log some statistics
    non_empty_tags = df['tags'].str.len() > 0
    logger.info(f"Keywords organized. {non_empty_tags.sum()}/{len(df)} products have non-empty tags")
    
    return df


def preprocess_data(file_path):
    """
    Complete preprocessing pipeline: load data, organize keywords, and apply stemming.
    
    Args:
        file_path (str): Path to the processed CSV file
        
    Returns:
        pd.DataFrame: Preprocessed dataframe with 'tags' column ready for vectorization
    """
    logger.info(f"Starting preprocessing pipeline for {file_path}")
    # Load data
    df = load_data(file_path)
    
    # Organize keywords and apply stemming
    df = organize_keywords(df)
    
    logger.info("Preprocessing pipeline completed successfully")
    return df
