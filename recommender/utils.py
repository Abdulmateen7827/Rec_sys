# Utility functions

import pandas as pd
import numpy as np
from .tfidf_recommender import TFIDFRecommender
import sys
from pathlib import Path

# Add parent directory to path for logger import
sys.path.append(str(Path(__file__).parent.parent))
from logger_config import setup_logger

# Setup logger
logger = setup_logger("utils")


def find_product_index(df, product_name):
    """
    Find the index of a product by name (case-insensitive partial match).
    
    Args:
        df (pd.DataFrame): Product dataframe
        product_name (str): Name or partial name of the product
        
    Returns:
        int or None: Index of the product if found, None otherwise
    """
    logger.debug(f"Searching for product: {product_name}")
    matches = df[df['name'].str.contains(product_name, case=False, na=False, regex=False)]
    
    if matches.empty:
        logger.warning(f"No matches found for product name: {product_name}")
        return None
    
    product_index = matches.index[0]
    logger.debug(f"Found product at index: {product_index}")
    return product_index


def get_product_list(df, limit=None):
    """
    Get a list of all product names for selection in the UI.
    
    Args:
        df (pd.DataFrame): Product dataframe
        limit (int, optional): Limit the number of products returned
        
    Returns:
        list: List of product names
    """
    logger.debug(f"Getting product list, limit={limit}")
    products = df['name'].tolist()
    
    if limit:
        logger.debug(f"Returning {limit} products (limited)")
        return products[:limit]
    
    logger.debug(f"Returning all {len(products)} products")
    return products


def format_recommendations(recommendations_df, include_score=True):
    """
    Format recommendations for display in the UI.
    
    Args:
        recommendations_df (pd.DataFrame): Recommendations dataframe
        include_score (bool): Whether to include similarity scores
        
    Returns:
        list: List of dictionaries with formatted recommendation data
    """
    logger.debug(f"Formatting {len(recommendations_df)} recommendations, include_score={include_score}")
    formatted = []
    
    for idx, row in recommendations_df.iterrows():
        rec = {
            'name': row['name'],
            'image': row.get('image', ''),
            'link': row.get('link', ''),
            'ratings': row.get('ratings', 'N/A'),
            'no_of_ratings': row.get('no_of_ratings', 'N/A'),
            'discount_price': row.get('discount_price', 'N/A'),
            'actual_price': row.get('actual_price', 'N/A')
        }
        
        if include_score and 'similarity_score' in row:
            rec['similarity_score'] = f"{row['similarity_score']:.3f}"
        
        formatted.append(rec)
    
    logger.debug(f"Formatted {len(formatted)} recommendations")
    return formatted


def load_recommender_system(model_path, data_file=None):
    """
    Load a pre-trained recommender system or build a new one.
    
    Args:
        model_path (str): Path to the saved model file
        data_file (str, optional): Path to data file if model doesn't exist
        
    Returns:
        tuple: (TFIDFRecommender, similarity_matrix, df)
    """
    import os
    
    logger.info(f"Loading recommender system. Model path: {model_path}, Data file: {data_file}")
    
    if os.path.exists(model_path):
        logger.info(f"Model file exists, loading from {model_path}")
        # Load existing model
        recommender = TFIDFRecommender()
        recommender.load_model(model_path)
        
        # Recompute similarity matrix
        logger.info("Recomputing similarity matrix")
        similarity_matrix = recommender.compute_similarity_matrix()
        
        logger.info("Recommender system loaded successfully")
        return recommender, similarity_matrix, recommender.df
    else:
        logger.info(f"Model file not found at {model_path}, building new model")
        # Build new model
        if data_file is None:
            logger.error("Model file not found and no data file provided")
            raise ValueError("Model file not found and no data file provided.")
        
        from .tfidf_recommender import build_recommender
        
        recommender, similarity_matrix = build_recommender(data_file, model_path)
        
        logger.info("Recommender system built successfully")
        return recommender, similarity_matrix, recommender.df
