# Configuration settings

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data files
DATA_FILE = str(DATA_DIR / "processed.csv")
MODEL_FILE = str(MODELS_DIR / "recommender_model.pkl")

# Recommender system parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
DEFAULT_TOP_N = 5

# Logging configuration
LOG_LEVEL = "INFO"
LOG_DIR = str(LOGS_DIR)

