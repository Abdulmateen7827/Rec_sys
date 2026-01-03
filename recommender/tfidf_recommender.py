# TF-IDF and Cosine Similarity Recommender

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import sys
from pathlib import Path

# Add parent directory to path for logger import
sys.path.append(str(Path(__file__).parent.parent))
from logger_config import setup_logger

# Setup logger
logger = setup_logger("tfidf_recommender")


class TFIDFRecommender:
    """
    TF-IDF based recommender system using cosine similarity.
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the TF-IDF Recommender.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF vectorizer
            ngram_range (tuple): Range of n-grams to consider (min, max)
        """
        logger.info(f"Initializing TFIDFRecommender with max_features={max_features}, ngram_range={ngram_range}")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.df = None
        self.feature_names = None
        logger.debug("TFIDFRecommender initialized successfully")
        
    def fit(self, df, text_column='tags'):
        """
        Fit the TF-IDF vectorizer on the processed text data.
        
        Args:
            df (pd.DataFrame): Dataframe with processed text in 'tags' column
            text_column (str): Name of the column containing processed text
        """
        logger.info(f"Fitting TF-IDF vectorizer on dataframe with shape {df.shape}, text_column: {text_column}")
        self.df = df.copy()
        
        # Extract text data
        text_data = self.df[text_column].fillna('').tolist()
        logger.debug(f"Extracted {len(text_data)} text samples")
        
        # Fit and transform the text data
        logger.info("Fitting and transforming text data with TF-IDF vectorizer")
        self.tfidf_matrix = self.vectorizer.fit_transform(text_data)
        
        # Get feature names
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        logger.info(f"Number of features: {len(self.feature_names)}")
        logger.info("TF-IDF vectorizer fitted successfully")
        
    def compute_similarity_matrix(self):
        """
        Compute cosine similarity matrix for all products.
        
        Returns:
            np.ndarray: Cosine similarity matrix of shape (n_products, n_products)
        """
        if self.tfidf_matrix is None:
            logger.error("Attempted to compute similarity matrix before fitting model")
            raise ValueError("Model must be fitted before computing similarity matrix.")
        
        logger.info("Computing cosine similarity matrix")
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(self.tfidf_matrix)
        logger.info(f"Similarity matrix computed with shape: {similarity_matrix.shape}")
        
        return similarity_matrix
    
    def get_recommendations(self, product_index, similarity_matrix, top_n=5):
        """
        Get top N recommendations for a given product index.
        
        Args:
            product_index (int): Index of the product in the dataframe
            similarity_matrix (np.ndarray): Precomputed similarity matrix
            top_n (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Top N recommended products with similarity scores
        """
        logger.info(f"Getting recommendations for product_index={product_index}, top_n={top_n}")
        if self.df is None:
            logger.error("Attempted to get recommendations before fitting model")
            raise ValueError("Model must be fitted before getting recommendations.")
        
        # Get similarity scores for the product
        similarity_scores = similarity_matrix[product_index]
        logger.debug(f"Retrieved similarity scores for product {product_index}")
        
        # Get top N similar products (excluding the product itself)
        similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
        similar_scores = similarity_scores[similar_indices]
        logger.debug(f"Top {len(similar_indices)} similar product indices: {similar_indices}")
        logger.debug(f"Similarity scores: {similar_scores}")
        
        # Create recommendations dataframe
        recommendations = self.df.iloc[similar_indices].copy()
        recommendations['similarity_score'] = similar_scores
        
        logger.info(f"Successfully generated {len(recommendations)} recommendations")
        return recommendations
    
    def get_recommendations_by_name(self, product_name, similarity_matrix, top_n=5):
        """
        Get recommendations for a product by its name.
        
        Args:
            product_name (str): Name of the product
            similarity_matrix (np.ndarray): Precomputed similarity matrix
            top_n (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Top N recommended products with similarity scores
        """
        logger.info(f"Getting recommendations by name: {product_name}, top_n={top_n}")
        if self.df is None:
            logger.error("Attempted to get recommendations before fitting model")
            raise ValueError("Model must be fitted before getting recommendations.")
        
        # Find product index by name
        matches = self.df[self.df['name'].str.contains(product_name, case=False, na=False, regex=False)]
        logger.debug(f"Found {len(matches)} matches for product name: {product_name}")
        
        if matches.empty:
            logger.warning(f"Product '{product_name}' not found in the dataset")
            raise ValueError(f"Product '{product_name}' not found in the dataset.")
        
        # Use the first match
        product_index = matches.index[0]
        logger.info(f"Using product at index {product_index} for recommendations")
        
        return self.get_recommendations(product_index, similarity_matrix, top_n)
    
    def save_model(self, filepath):
        """
        Save the trained model and vectorizer to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        logger.info(f"Saving model to {filepath}")
        model_data = {
            'vectorizer': self.vectorizer,
            'df': self.df,
            'tfidf_matrix': self.tfidf_matrix,
            'feature_names': self.feature_names
        }
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
            logger.info(f"Model saved successfully to {filepath} (size: {file_size:.2f} MB)")
        except Exception as e:
            logger.error(f"Error saving model to {filepath}: {e}", exc_info=True)
            raise
    
    def load_model(self, filepath):
        """
        Load a trained model and vectorizer from disk.
        
        Args:
            filepath (str): Path to load the model from
        """
        logger.info(f"Loading model from {filepath}")
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found at {filepath}")
                raise FileNotFoundError(f"Model file not found at {filepath}")
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.df = model_data['df']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.feature_names = model_data['feature_names']
            
            logger.info(f"Model loaded successfully from {filepath}")
            logger.info(f"Loaded dataframe shape: {self.df.shape}")
            logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {e}", exc_info=True)
            raise


def build_recommender(data_file, model_save_path=None):
    """
    Build and train a TF-IDF recommender system.
    
    Args:
        data_file (str): Path to processed CSV file
        model_save_path (str, optional): Path to save the trained model
        
    Returns:
        tuple: (TFIDFRecommender, similarity_matrix)
    """
    logger.info(f"Building recommender system from data file: {data_file}")
    from .data_processor import preprocess_data
    
    # Preprocess data
    logger.info("Loading and preprocessing data...")
    df = preprocess_data(data_file)
    logger.info(f"Data preprocessed. Shape: {df.shape}")
    
    # Initialize recommender
    logger.info("Initializing TFIDFRecommender")
    recommender = TFIDFRecommender()
    
    # Fit the model
    logger.info("Fitting TF-IDF vectorizer...")
    recommender.fit(df)
    
    # Compute similarity matrix
    logger.info("Computing cosine similarity matrix...")
    similarity_matrix = recommender.compute_similarity_matrix()
    
    # Save model if path provided
    if model_save_path:
        logger.info(f"Saving model to {model_save_path}")
        recommender.save_model(model_save_path)
    
    logger.info("Recommender system built successfully!")
    
    return recommender, similarity_matrix
