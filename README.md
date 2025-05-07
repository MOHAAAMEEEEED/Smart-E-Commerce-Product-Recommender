# Smart E-Commerce Product Recommender

## Description
A hybrid recommender system that suggests products based on both collaborative filtering and content-based filtering techniques. This system analyzes user behavior and product attributes to provide personalized product recommendations for e-commerce applications.

## Concepts Used
- Collaborative filtering: Recommends products based on user similarities
- Content-based filtering: Recommends products based on product attributes
- Hybrid recommendation models: Combines multiple recommendation strategies
- Ensemble-based methods: Uses weighted combination and cascade techniques

## Project Structure
```
smart_recommender/
│
├── 📁 data/
│   └── amazon.csv                   # E-commerce product dataset
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb    # Data analysis and exploration
│   ├── 02_collaborative_filtering.ipynb # Collaborative filtering analysis
│   ├── 03_content_based_filtering.ipynb # Content-based filtering analysis
│   └── 04_hybrid_modeling.ipynb     # Hybrid model development
│
├── 📁 src/
│   ├── __init__.py
│   ├── load_data.py                 # Data loading and preprocessing
│   │
│   ├── 📁 models/
│   │   ├── train_collaborative.py   # Collaborative model training
│   │   ├── train_content.py         # Content-based model training
│   │   └── train_hybrid.py          # Hybrid model training
│   │
│   ├── 📁 recommenders/
│   │   ├── collaborative.py         # Collaborative recommender
│   │   ├── content_based.py         # Content-based recommender
│   │   └── hybrid.py                # Hybrid recommender
│   │
│   └── 📁 utils/
│       ├── metrics.py               # Evaluation metrics
│       └── visualization.py         # Data and results visualization
│
└── README.md
```

## Installation

# Requirements
```bash
pip install -r requirements.txt
``

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/smart-e-commerce-recommender.git
cd smart-e-commerce-recommender

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
Place your e-commerce data in the `data/` directory. The system expects a CSV file with at least the following columns:
- `user_id`: Unique identifier for users
- `product_id`: Unique identifier for products
- `rating`: Numeric rating or interaction strength

You can also include optional columns like:
- `timestamp`: When the interaction occurred
- `title`: Product title
- `description`: Product description
- `category`: Product category

### 2. Data Exploration
```bash
# Run the data exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 3. Train and Evaluate Models

#### Train Collaborative Filtering Model
```python
from src.models.train_collaborative import train_collaborative_model, prepare_surprise_data
from src.load_data import load_amazon_data, preprocess_data

# Load and preprocess data
df = load_amazon_data()
df = preprocess_data(df)

# Prepare data for Surprise
data = prepare_surprise_data(df)

# Train the model
model, metrics = train_collaborative_model(data, save_path='models/collaborative_model.pkl')
```

#### Train Content-Based Model
```python
from src.models.train_content import create_item_features, build_content_model, compute_similarity_matrix

# Create item features
item_features = create_item_features(df)

# Build content model
vectorizer, item_matrix, item_ids = build_content_model(
    item_features, 
    save_path='models/content_model'
)

# Compute similarity matrix
sim_matrix = compute_similarity_matrix(item_matrix, save_path='models/content_model')
```

#### Train Hybrid Model
```python
from src.models.train_hybrid import train_hybrid_model

# Train hybrid model
hybrid_model = train_hybrid_model(
    df=df,
    collab_model_path='models/collaborative_model.pkl',
    content_model_path='models/content_model',
    strategy='weighted',  # Options: 'weighted', 'feature', 'switch', 'cascade'
    save_path='models/hybrid_model'
)
```

### 4. Make Recommendations

#### Collaborative Recommendations
```python
from src.recommenders.collaborative import CollaborativeRecommender

# Initialize recommender
recommender = CollaborativeRecommender('models/collaborative_model.pkl')

# Get recommendations for a user
recommendations = recommender.recommend_items(user_id='user123', n_recommendations=5)
```

#### Content-Based Recommendations
```python
from src.recommenders.content_based import ContentBasedRecommender

# Initialize recommender
recommender = ContentBasedRecommender(
    vectorizer_path='models/content_model_vectorizer.pkl',
    item_matrix_path='models/content_model_item_matrix.pkl',
    similarity_matrix_path='models/content_model_similarity.pkl',
    item_ids_path='models/content_model_item_ids.pkl'
)

# Get similar items
similar_items = recommender.get_similar_items(item_id='product456')

# Get recommendations based on user history
user_history = {'product1': 5.0, 'product2': 4.0}  # Dict of {item_id: rating}
recommendations = recommender.recommend_for_user_history(user_history)
```

#### Hybrid Recommendations
```python
from src.recommenders.hybrid import HybridRecommender

# Initialize recommender
recommender = HybridRecommender(
    collab_model_path='models/collaborative_model.pkl',
    content_model_path='models/content_model',
    strategy='weighted'  # Options: 'weighted', 'switch', 'cascade'
)

# Get recommendations for a user
user_history = {'product1': 5.0, 'product2': 4.0}
recommendations = recommender.recommend_items(
    user_id='user123',
    user_history=user_history,
    n_recommendations=5
)

# Get explanation for a recommendation
explanation = recommender.explain_recommendation(
    user_id='user123',
    item_id='product456',
    user_history=user_history
)
```

### 5. Evaluate Models
```python
from src.utils.metrics import evaluate_recommender
from src.load_data import load_amazon_data, split_train_test

# Load data and split into train/test
df = load_amazon_data()
train_df, test_df = split_train_test(df)

# Evaluate recommender
metrics = evaluate_recommender(recommender, test_df)
```

### 6. Visualize Results
```python
from src.utils.visualization import plot_metrics_comparison, create_visualizations_for_data

# Create visualizations for your data
create_visualizations_for_data(df)

# Compare model performances
metrics_by_model = {
    'Collaborative': collab_metrics,
    'Content-based': content_metrics,
    'Hybrid': hybrid_metrics
}
plot_metrics_comparison(metrics_by_model)
```

## Hybrid Strategies

The system supports multiple hybrid recommendation strategies:

### 1. Weighted
Combines scores from both models using weighted average. Default weights are 0.7 for collaborative and 0.3 for content-based.

### 2. Switching
Dynamically switches between collaborative and content-based recommendations based on certain conditions.

### 3. Cascade
Uses a sequential approach where one recommender filters candidates and the other re-ranks them.

### 4. Feature-based
Treats recommendations from individual models as features and uses a meta-model to make the final prediction.

## Metrics

The system evaluates recommendations using standard metrics:

- RMSE (Root Mean Square Error)
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Coverage
- Diversity
- Novelty

## License
MIT
