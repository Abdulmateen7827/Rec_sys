# Main Streamlit application

import streamlit as st
import pandas as pd
from recommender import (
    build_recommender,
    load_recommender_system,
    find_product_index,
    format_recommendations
)
import os
from logger_config import setup_logger
import config

# Setup logger
logger = setup_logger("app", config.LOG_DIR)

# Page configuration
st.set_page_config(
    page_title="A baseline item-to-item Product Recommender System",
    layout="wide"
)

# Title
st.title(" A baseline item-to-item Product Recommender System")
st.markdown("---")

# Configuration
DATA_FILE = config.DATA_FILE
MODEL_FILE = config.MODEL_FILE

logger.info(f"Initializing app with DATA_FILE: {DATA_FILE}, MODEL_FILE: {MODEL_FILE}")

# Initialize session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
    st.session_state.similarity_matrix = None
    st.session_state.df = None
    st.session_state.recommendations = None
    st.session_state.clicked_product = None
    st.session_state.recommendation_history = []  # Track exploration path
    logger.info("Initialized session state")

# Load or build recommender system
@st.cache_resource
def load_system():
    """Load or build the recommender system."""
    logger.info("Starting to load or build recommender system")
    os.makedirs("models", exist_ok=True)
    
    if os.path.exists(MODEL_FILE):
        logger.info(f"Model file exists at {MODEL_FILE}, attempting to load")
        try:
            recommender, similarity_matrix, df = load_recommender_system(
                MODEL_FILE, 
                DATA_FILE
            )
            logger.info(f"Successfully loaded model. DataFrame shape: {df.shape}")
            return recommender, similarity_matrix, df
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            st.error(f"Error loading model: {e}")
            return None, None, None
    else:
        logger.info(f"Model file not found at {MODEL_FILE}, building new model")
        try:
            recommender, similarity_matrix = build_recommender(
                DATA_FILE,
                MODEL_FILE
            )
            logger.info(f"Successfully built model. DataFrame shape: {recommender.df.shape}")
            return recommender, similarity_matrix, recommender.df
        except Exception as e:
            logger.error(f"Error building model: {e}", exc_info=True)
            st.error(f"Error building model: {e}")
            return None, None, None

# Load the system
with st.spinner("Loading recommender system..."):
    if st.session_state.recommender is None:
        logger.info("Loading recommender system into session state")
        recommender, similarity_matrix, df = load_system()
        if recommender is not None:
            st.session_state.recommender = recommender
            st.session_state.similarity_matrix = similarity_matrix
            st.session_state.df = df
            logger.info("Recommender system loaded successfully into session state")
            st.success("Recommender system loaded successfully!")
        else:
            logger.error("Failed to load recommender system")

# Main interface
if st.session_state.recommender is not None and st.session_state.df is not None:
    df = st.session_state.df
    recommender = st.session_state.recommender
    similarity_matrix = st.session_state.similarity_matrix
    
    # Sidebar for product selection
    st.sidebar.header("🔍 Product Selection")
    
    # Handle clicked product from recommendations
    product_names = df['name'].tolist()
    
    # Check if a product was clicked (before we process it)
    product_was_clicked = st.session_state.clicked_product is not None
    
    if product_was_clicked:
        # Use clicked product
        selected_product = st.session_state.clicked_product
        # Find index for selectbox
        try:
            clicked_index = product_names.index(selected_product)
        except ValueError:
            clicked_index = 0
        logger.info(f"Product clicked from recommendations: {selected_product}")
    else:
        # Get current selection from session state or use first product
        current_index = 0
        if 'current_product' in st.session_state:
            try:
                current_index = product_names.index(st.session_state.current_product)
            except ValueError:
                current_index = 0
    
    # Display selectbox (will show clicked product if one was clicked)
    if product_was_clicked:
        selected_product = st.sidebar.selectbox(
            "Select a product:",
            product_names,
            index=clicked_index,
            key="product_selectbox"
        )
        # Clear clicked product after using
        st.session_state.clicked_product = None
    else:
        selected_product = st.sidebar.selectbox(
            "Select a product:",
            product_names,
            index=current_index,
            key="product_selectbox"
        )
    
    # Track if product changed
    product_changed = (
        'current_product' not in st.session_state or 
        st.session_state.current_product != selected_product
    )
    
    # Update current product in session state
    st.session_state.current_product = selected_product
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider(
        "Number of recommendations:",
        min_value=1,
        max_value=10,
        value=5
    )
    
    # Auto-generate recommendations function
    def generate_recommendations(product_name, top_n):
        """Generate recommendations for a given product."""
        try:
            product_index = find_product_index(df, product_name)
            logger.debug(f"Product index found: {product_index}")
            
            if product_index is not None:
                recommendations = recommender.get_recommendations(
                    product_index,
                    similarity_matrix,
                    top_n=top_n
                )
                logger.info(f"Generated {len(recommendations)} recommendations for {product_name}")
                st.session_state.recommendations = recommendations
                
                # Add to history
                if product_name not in st.session_state.recommendation_history:
                    st.session_state.recommendation_history.append(product_name)
                return True
            else:
                logger.warning(f"Product not found: {product_name}")
                return False
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}", exc_info=True)
            return False
    
    # Auto-generate recommendations on product change, click, or initial load
    should_auto_generate = (
        st.session_state.recommendations is None or 
        product_changed or 
        product_was_clicked
    )
    
    if should_auto_generate:
        logger.info(f"Auto-generating recommendations for: {selected_product}")
        if not generate_recommendations(selected_product, num_recommendations):
            st.error("Product not found!")
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Recommendations", type="secondary"):
        logger.info(f"User manually refreshed recommendations for: {selected_product}")
        if not generate_recommendations(selected_product, num_recommendations):
            st.error("Product not found!")
    
    # Show exploration history
    if len(st.session_state.recommendation_history) > 1:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📍 Exploration Path")
        history_text = " → ".join(st.session_state.recommendation_history[-5:])  # Show last 5
        st.sidebar.caption(history_text)
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.recommendation_history = []
            logger.info("User cleared recommendation history")
    
    # Display selected product
    st.header("📦 Selected Product")
    selected_idx = find_product_index(df, selected_product)
    logger.debug(f"Displaying product at index: {selected_idx}")
    if selected_idx is not None:
        selected_row = df.iloc[selected_idx]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if pd.notna(selected_row.get('image')):
                st.image(selected_row['image'], width=200)
        
        with col2:
            st.subheader(selected_row['name'])
            st.write(f"**Ratings:** {selected_row.get('ratings', 'N/A')}")
            st.write(f"**Number of Ratings:** {selected_row.get('no_of_ratings', 'N/A')}")
            st.write(f"**Discount Price:** {selected_row.get('discount_price', 'N/A')}")
            st.write(f"**Actual Price:** {selected_row.get('actual_price', 'N/A')}")
            if pd.notna(selected_row.get('link')):
                st.markdown(f"[View on Amazon]({selected_row['link']})")
    
    # Display recommendations
    if 'recommendations' in st.session_state and st.session_state.recommendations is not None:
        st.markdown("---")
        st.header("🎯 Recommended Products")
        st.caption("💡 Click on any recommended product to explore similar items!")
        
        recommendations = st.session_state.recommendations
        formatted_recs = format_recommendations(recommendations, include_score=True)
        logger.info(f"Displaying {len(formatted_recs)} formatted recommendations")
        
        # Display recommendations in a grid
        num_cols = min(3, len(formatted_recs))
        cols = st.columns(num_cols)
        
        for idx, rec in enumerate(formatted_recs):
            col_idx = idx % num_cols
            with cols[col_idx]:
                # Product card container
                with st.container():
                    if rec['image']:
                        st.image(rec['image'], width=150, use_container_width=True)
                    
                    st.markdown(f"**{rec['name']}**")
                    st.caption(f"🔗 Similarity: {rec['similarity_score']}")
                    st.write(f"⭐ {rec['ratings']} ({rec['no_of_ratings']} ratings)")
                    st.write(f"💰 {rec['discount_price']}")
                    
                    # Clickable button to explore this product
                    if st.button(
                        "🔍 Explore Similar Products",
                        key=f"explore_{idx}_{hash(rec['name']) % 10000}",
                        use_container_width=True,
                        type="primary"
                    ):
                        st.session_state.clicked_product = rec['name']
                        logger.info(f"User clicked to explore: {rec['name']}")
                        st.rerun()
                    
                    # External link
                    if rec['link']:
                        st.markdown(f"[🔗 View on Amazon]({rec['link']})", unsafe_allow_html=True)
                    
                    st.markdown("---")
else:
    logger.error("Failed to load recommender system - recommender or df is None")
    st.error("Failed to load recommender system. Please check the data file and try again.")
