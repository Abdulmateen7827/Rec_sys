# 🔌 Content-Based Recommender System

A simple content-based recommender system prototype built with TF-IDF vectorization and cosine similarity. This system recommends products based on their textual features (product names, descriptions) using natural language processing techniques.

## ✨ Features

- **Content-Based Filtering**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to extract features from product text
- **Cosine Similarity**: Finds similar products based on text similarity
- **Interactive Streamlit UI**: Beautiful web interface for exploring recommendations
- **Click-to-Explore**: Click on any recommended product to discover similar items in an exploration loop
- **Comprehensive Logging**: Track all operations with detailed logs saved to files
- **Model Persistence**: Save and load trained models for faster startup

## 🎯 How It Works

1. **Text Preprocessing**: Product names are cleaned, stemmed, and stopwords are removed
2. **TF-IDF Vectorization**: Text is converted to numerical vectors using TF-IDF
3. **Similarity Computation**: Cosine similarity matrix is computed for all products
4. **Recommendation Generation**: Top N most similar products are returned based on cosine similarity scores

## 📋 Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- nltk >= 3.8
- streamlit >= 1.28.0

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rec_sys
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (automatically handled on first run):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 💻 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will:
- Load or build the recommender model
- Display an interactive interface
- Allow you to select products and get recommendations
- Enable click-through exploration of similar products

### Using the Recommender Programmatically

```python
from recommender import build_recommender, load_recommender_system

# Build a new recommender
recommender, similarity_matrix = build_recommender(
    data_file="data/processed.csv",
    model_save_path="models/recommender_model.pkl"
)

# Get recommendations for a product by index
recommendations = recommender.get_recommendations(
    product_index=0,
    similarity_matrix=similarity_matrix,
    top_n=5
)
```

## 📁 Project Structure

```
rec_sys/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── logger_config.py                # Logging configuration
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── data/                           # Data directory
│   └── processed.csv              # Processed product data
├── models/                         # Saved models directory
│   └── recommender_model.pkl      # Trained model (auto-generated)
├── logs/                          # Log files directory
│   ├── app_YYYYMMDD.log
│   ├── tfidf_recommender_YYYYMMDD.log
│   ├── data_processor_YYYYMMDD.log
│   └── utils_YYYYMMDD.log
└── recommender/                    # Recommender package
    ├── __init__.py
    ├── tfidf_recommender.py       # TF-IDF recommender class
    ├── data_processor.py          # Data loading and preprocessing
    └── utils.py                  # Utility functions
```

## 🔧 Configuration

Edit `config.py` to customize:
- Data file paths
- Model save location
- TF-IDF parameters (max_features, ngram_range)
- Logging settings

## 📊 Data Format

The input CSV file should contain at minimum:
- `name`: Product name (used for recommendations)
- Additional columns like `ratings`, `image`, `link`, etc. are optional but enhance the UI

## 🎨 Features in Detail

### Interactive Exploration
- Click "🔍 Explore Similar Products" on any recommendation
- Automatically loads new recommendations for the clicked product
- Track your exploration path in the sidebar
- Create an endless discovery loop

### Logging
- All operations are logged with timestamps
- Separate log files for each module
- Logs saved daily in `logs/` directory
- Includes error tracking with stack traces

### Model Management
- Models are automatically saved after training
- Faster startup by loading pre-trained models
- Automatic model rebuilding if file is missing

## 🛠️ Technologies Used

- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **NLTK**: Text preprocessing (stemming, stopword removal)
- **Streamlit**: Interactive web interface
- **pandas**: Data manipulation
- **numpy**: Numerical computations

