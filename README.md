# E-commerce Recommendation System

A robust, end-to-end recommendation system for e-commerce applications, featuring multiple recommendation strategies and detailed evaluation. This project is based on the accompanying Jupyter notebook, which demonstrates the full workflow from data preparation to model evaluation and visualization.

## Project Introduction

This project aims to provide a comprehensive recommendation system for e-commerce platforms, leveraging both content and user behavior to generate relevant product suggestions. The system is designed to be modular, allowing for experimentation with different recommendation algorithms and evaluation techniques. The included notebook walks through the entire process, including:
- Data loading and preprocessing
- Implementation of various recommendation strategies
- Evaluation of recommendation quality
- Visualization of results

## Data Preparation

The system expects a product dataset in TSV format with columns such as:
- Uniq Id
- Product Id
- Product Rating
- Product Reviews Count
- Product Category
- Product Brand
- Product Name
- Product Image Url
- Product Description
- Product Tags

**Preprocessing steps include:**
- Loading the dataset into a pandas DataFrame
- Cleaning and normalizing text fields (e.g., product descriptions, tags)
- Handling missing values
- Generating synthetic user ratings data for collaborative filtering (if real user data is not available)

**Synthetic Data Generation:**
For collaborative filtering, the notebook can create a synthetic ratings dataset by randomly assigning ratings from synthetic users to products, simulating real-world user behavior.

## Recommendation Strategies

### 1. Content-Based Recommendations
**Concept:**
Recommends products similar to a given product by analyzing product features such as tags, description, and category. Uses NLP techniques (e.g., TF-IDF vectorization) to compute product similarity.

**Implementation:**
- Vectorizes product features using TF-IDF or similar methods
- Computes cosine similarity between products
- Returns top-N most similar products to a given item

**Example Usage:**
```python
recommend_products_content_based(item_name="OPI Nail Lacquer", product_df, n=5)
```
**Expected Output:**
A DataFrame of the top 5 products most similar to "OPI Nail Lacquer" with similarity scores.

### 2. Collaborative Filtering Recommendations
**Concept:**
Recommends products based on user behavior, leveraging the preferences of similar users. Uses matrix factorization (e.g., SVD) on the user-item ratings matrix.

**Implementation:**
- Builds a user-item ratings matrix (synthetic or real)
- Applies matrix factorization to learn latent user and product features
- Predicts ratings for unseen products for a given user
- Returns top-N recommended products for the user

**Example Usage:**
```python
recommend_products_collaborative(user_id="user_0043", product_df, n=5)
```
**Expected Output:**
A DataFrame of the top 5 recommended products for the specified user, with predicted ratings.

### 3. Hybrid Recommendations
**Concept:**
Combines content-based and collaborative filtering approaches to leverage both product features and user behavior for more robust recommendations.

**Implementation:**
- Generates recommendations from both content-based and collaborative models
- Merges and ranks results (e.g., by weighted score or intersection)
- Returns top-N hybrid recommendations

**Example Usage:**
```python
recommend_products_hybrid(user_id="user_0043", item_name="OPI Nail Lacquer", product_df, n=5)
```
**Expected Output:**
A DataFrame of the top 5 hybrid recommendations for the user and item context.

### 4. Rating-Based (Trending) Recommendations
**Concept:**
Recommends trending products based on high ratings and review counts, independent of user or item context.

**Implementation:**
- Filters products by minimum rating and review count thresholds
- Sorts by rating and popularity
- Returns top-N trending products

**Example Usage:**
```python
recommend_products_rating_based(product_df, min_rating=4.0, min_reviews=5, n=5)
```
**Expected Output:**
A DataFrame of the top 5 trending products meeting the specified criteria.

## Evaluation

The system includes robust evaluation metrics to assess recommendation quality:
- **RMSE (Root Mean Squared Error):** Measures the average prediction error for ratings.
- **MAE (Mean Absolute Error):** Measures the average absolute difference between predicted and actual ratings.
- **Catalog Coverage:** Percentage of the product catalog that appears in recommendations.
- **Precision/Recall:** (If implemented) Measures the relevance and completeness of recommendations.

**Example Evaluation Output:**
```
Collaborative model evaluation:
  RMSE: 1.54
  MAE: 1.09
  Test samples: 45
  Catalog coverage (sample): 0.92%
```

## Example Outputs & Visualizations

The notebook provides sample outputs and visualizations, such as:
- Printed tables of recommended products with scores/ratings
- Plots of explained variance for matrix factorization models
- Evaluation metric summaries


## Notebook Usage

To run the notebook:
1. Open `e_commerce_reommendation.ipynb` in Jupyter Notebook or JupyterLab.
2. Run all cells sequentially:
   - Data loading and preprocessing
   - Synthetic data generation (if needed)
   - Model training and recommendation generation
   - Evaluation and visualization
3. Modify parameters (e.g., user_id, item_name, top_n) in the relevant cells to experiment with different scenarios.
4. Review printed outputs and plots for insights.

## Acknowledgments

- Dataset source: [Walmart.com Product Reviews](https://www.kaggle.com/datasets/promptcloud/walmart-product-review-dataset)

