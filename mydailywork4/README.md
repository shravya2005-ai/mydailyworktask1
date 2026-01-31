# Movie Recommendation System

A simple recommendation system that demonstrates three different approaches to generating personalized recommendations:

## Features

### 1. Collaborative Filtering
- Finds users with similar preferences
- Recommends items liked by similar users
- Uses cosine similarity to measure user similarity

### 2. Content-Based Filtering
- Analyzes item features (genres, descriptions)
- Recommends similar items based on content
- Uses TF-IDF vectorization and cosine similarity

### 3. Hybrid Approach
- Combines collaborative and content-based methods
- Weighted scoring system (70% collaborative, 30% content-based)
- Provides more robust recommendations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the demo:
```bash
python recommendation_system.py
```

## Example Usage in Code

```python
from recommendation_system import RecommendationSystem

# Initialize system
rec_system = RecommendationSystem()
rec_system.load_sample_data()
rec_system.build_content_similarity()

# Get collaborative filtering recommendations
collab_recs = rec_system.collaborative_filtering_recommendations(user_id=1)

# Get content-based recommendations
content_recs = rec_system.content_based_recommendations(movie_id=1)

# Get hybrid recommendations
hybrid_recs = rec_system.hybrid_recommendations(user_id=1)

# Add new rating
rec_system.add_user_rating(user_id=1, movie_id=3, rating=5)
```

## System Architecture

The recommendation system includes:
- **Sample Data**: 10 movies with genres and descriptions, 20 user ratings
- **User-Item Matrix**: For collaborative filtering calculations
- **Content Similarity Matrix**: For content-based recommendations
- **Hybrid Scoring**: Combines multiple recommendation approaches

## Extending the System

To use with your own data:
1. Replace the sample data in `load_sample_data()` method
2. Ensure your data has the required columns: `user_id`, `movie_id`, `rating`, `title`, `genres`, `description`
3. The system will automatically build the necessary matrices for recommendations