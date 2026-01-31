import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json

class RecommendationSystem:
    def __init__(self):
        self.users_df = None
        self.movies_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.content_similarity_matrix = None
        
    def load_sample_data(self):
        """Load sample movie data for demonstration"""
        # Sample movies with genres and descriptions
        movies_data = {
            'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'title': [
                'The Matrix', 'Inception', 'Titanic', 'The Godfather',
                'Pulp Fiction', 'Forrest Gump', 'The Dark Knight',
                'Interstellar', 'Casablanca', 'Goodfellas'
            ],
            'genres': [
                'Action,Sci-Fi', 'Action,Sci-Fi,Thriller', 'Romance,Drama',
                'Crime,Drama', 'Crime,Drama', 'Drama,Romance',
                'Action,Crime,Drama', 'Sci-Fi,Drama', 'Romance,Drama',
                'Crime,Drama'
            ],
            'description': [
                'A computer hacker learns about the true nature of reality',
                'A thief enters peoples dreams to steal secrets',
                'A romance aboard the ill-fated RMS Titanic',
                'The aging patriarch of a crime dynasty transfers control',
                'The lives of two mob hitmen, a boxer, and others intertwine',
                'The presidencies of Kennedy and Johnson through Vietnam',
                'Batman fights crime in Gotham City against the Joker',
                'A team of explorers travel through a wormhole in space',
                'A cynical nightclub owner aids his former lover',
                'The story of Henry Hill and his life in the mob'
            ]
        }
        
        # Sample user ratings
        ratings_data = {
            'user_id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
            'movie_id': [1, 2, 7, 8, 1, 3, 6, 9, 2, 4, 5, 10, 3, 6, 7, 9, 4, 5, 8, 10],
            'rating': [5, 4, 5, 4, 4, 5, 4, 5, 5, 5, 4, 5, 4, 5, 3, 4, 5, 5, 3, 4]
        }
        
        self.movies_df = pd.DataFrame(movies_data)
        self.ratings_df = pd.DataFrame(ratings_data)
        
        # Create user-item matrix for collaborative filtering
        self.user_item_matrix = self.ratings_df.pivot(
            index='user_id', 
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        
        print("Sample data loaded successfully!")
        print(f"Movies: {len(self.movies_df)}")
        print(f"Ratings: {len(self.ratings_df)}")
        
    def build_content_similarity(self):
        """Build content-based similarity matrix using movie genres and descriptions"""
        # Combine genres and descriptions for content analysis
        self.movies_df['content'] = self.movies_df['genres'] + ' ' + self.movies_df['description']
        
        # Create TF-IDF vectors
        tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = tfidf.fit_transform(self.movies_df['content'])
        
        # Calculate cosine similarity
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print("Content-based similarity matrix built!")
    def collaborative_filtering_recommendations(self, user_id, num_recommendations=5):
        """Generate recommendations using collaborative filtering"""
        if user_id not in self.user_item_matrix.index:
            return f"User {user_id} not found in the system"
        
        # Calculate user similarity using cosine similarity
        user_similarity = cosine_similarity(self.user_item_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity, 
            index=self.user_item_matrix.index, 
            columns=self.user_item_matrix.index
        )
        
        # Get similar users (excluding the user themselves)
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
        
        # Get movies rated by the target user
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Calculate weighted ratings for unrated movies
        recommendations = {}
        for movie_id in unrated_movies:
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user_id in similar_users.index[:3]:  # Top 3 similar users
                if self.user_item_matrix.loc[similar_user_id, movie_id] > 0:
                    similarity_score = similar_users[similar_user_id]
                    rating = self.user_item_matrix.loc[similar_user_id, movie_id]
                    weighted_sum += similarity_score * rating
                    similarity_sum += similarity_score
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                recommendations[movie_id] = predicted_rating
        
        # Sort and get top recommendations
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        
        # Get movie details
        result = []
        for movie_id, predicted_rating in top_recommendations:
            movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
            result.append({
                'movie_id': int(movie_id),  # Convert to Python int
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'predicted_rating': round(float(predicted_rating), 2)  # Convert to Python float
            })
        
        return result
    
    def content_based_recommendations(self, movie_id, num_recommendations=5):
        """Generate recommendations using content-based filtering"""
        if movie_id not in self.movies_df['movie_id'].values:
            return f"Movie {movie_id} not found in the system"
        
        # Get the index of the movie
        movie_idx = self.movies_df[self.movies_df['movie_id'] == movie_id].index[0]
        
        # Get similarity scores for this movie
        similarity_scores = list(enumerate(self.content_similarity_matrix[movie_idx]))
        
        # Sort movies by similarity (excluding the movie itself)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]
        
        # Get top similar movies
        top_similar = similarity_scores[:num_recommendations]
        
        result = []
        for idx, similarity_score in top_similar:
            movie_info = self.movies_df.iloc[idx]
            result.append({
                'movie_id': int(movie_info['movie_id']),  # Convert to Python int
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'similarity_score': round(float(similarity_score), 3)  # Convert to Python float
            })
        
        return result
    
    def hybrid_recommendations(self, user_id, num_recommendations=5):
        """Generate recommendations using hybrid approach (collaborative + content-based)"""
        # Get collaborative filtering recommendations
        collab_recs = self.collaborative_filtering_recommendations(user_id, num_recommendations * 2)
        
        if isinstance(collab_recs, str):  # Error message
            return collab_recs
        
        # Get user's favorite genres based on their ratings
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        high_rated_movies = user_ratings[user_ratings['rating'] >= 4]['movie_id'].tolist()
        
        # Combine content-based recommendations for user's favorite movies
        content_recs = []
        for movie_id in high_rated_movies[:2]:  # Use top 2 favorite movies
            movie_content_recs = self.content_based_recommendations(movie_id, 3)
            if isinstance(movie_content_recs, list):
                content_recs.extend(movie_content_recs)
        
        # Combine and deduplicate recommendations
        all_recs = {}
        
        # Add collaborative filtering recommendations with higher weight
        for rec in collab_recs:
            all_recs[rec['movie_id']] = {
                'title': rec['title'],
                'genres': rec['genres'],
                'score': rec['predicted_rating'] * 0.7  # 70% weight for collaborative
            }
        
        # Add content-based recommendations with lower weight
        for rec in content_recs:
            movie_id = rec['movie_id']
            if movie_id in all_recs:
                all_recs[movie_id]['score'] += rec['similarity_score'] * 5 * 0.3  # 30% weight for content
            else:
                all_recs[movie_id] = {
                    'title': rec['title'],
                    'genres': rec['genres'],
                    'score': rec['similarity_score'] * 5 * 0.3
                }
        
        # Sort by combined score and return top recommendations
        sorted_recs = sorted(all_recs.items(), key=lambda x: x[1]['score'], reverse=True)
        
        result = []
        for movie_id, rec_data in sorted_recs[:num_recommendations]:
            result.append({
                'movie_id': int(movie_id),  # Convert to Python int
                'title': rec_data['title'],
                'genres': rec_data['genres'],
                'hybrid_score': round(float(rec_data['score']), 2)  # Convert to Python float
            })
        
        return result
    def add_user_rating(self, user_id, movie_id, rating):
        """Add a new user rating to the system"""
        new_rating = pd.DataFrame({
            'user_id': [user_id],
            'movie_id': [movie_id],
            'rating': [rating]
        })
        
        self.ratings_df = pd.concat([self.ratings_df, new_rating], ignore_index=True)
        
        # Rebuild user-item matrix
        self.user_item_matrix = self.ratings_df.pivot(
            index='user_id', 
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        
        print(f"Added rating: User {user_id} rated Movie {movie_id} with {rating} stars")
    
    def get_user_profile(self, user_id):
        """Get user's rating history and preferences"""
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        if user_ratings.empty:
            return f"No ratings found for user {user_id}"
        
        # Merge with movie information
        user_profile = user_ratings.merge(self.movies_df, on='movie_id')
        
        # Calculate average rating and favorite genres
        avg_rating = user_profile['rating'].mean()
        
        # Get favorite genres
        all_genres = []
        for genres_str in user_profile['genres']:
            all_genres.extend(genres_str.split(','))
        
        genre_counts = pd.Series(all_genres).value_counts()
        
        return {
            'user_id': int(user_id),
            'total_ratings': int(len(user_profile)),
            'average_rating': round(float(avg_rating), 2),
            'favorite_genres': {k: int(v) for k, v in genre_counts.head(3).to_dict().items()},
            'rated_movies': user_profile[['title', 'rating', 'genres']].to_dict('records')
        }
    
    def display_recommendations(self, recommendations, rec_type="Recommendations"):
        """Display recommendations in a formatted way"""
        print(f"\n=== {rec_type} ===")
        if isinstance(recommendations, str):
            print(recommendations)
            return
        
        if not recommendations:
            print("No recommendations found.")
            return
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']} ({rec['genres']})")
            if 'predicted_rating' in rec:
                print(f"   Predicted Rating: {rec['predicted_rating']}/5")
            elif 'similarity_score' in rec:
                print(f"   Similarity Score: {rec['similarity_score']}")
            elif 'hybrid_score' in rec:
                print(f"   Hybrid Score: {rec['hybrid_score']}")
            print()


def main():
    """Demonstrate the recommendation system"""
    print("🎬 Movie Recommendation System Demo")
    print("=" * 40)
    
    # Initialize the system
    rec_system = RecommendationSystem()
    rec_system.load_sample_data()
    rec_system.build_content_similarity()
    
    print("\n📊 Sample Data Overview:")
    print("Movies in system:")
    for _, movie in rec_system.movies_df.iterrows():
        print(f"  {movie['movie_id']}. {movie['title']} ({movie['genres']})")
    
    # Show user profile
    print("\n👤 User Profile Example:")
    user_profile = rec_system.get_user_profile(1)
    print(f"User 1 Profile: {user_profile}")
    
    # Collaborative Filtering Demo
    print("\n🤝 Collaborative Filtering Recommendations for User 1:")
    collab_recs = rec_system.collaborative_filtering_recommendations(1)
    rec_system.display_recommendations(collab_recs, "Collaborative Filtering")
    
    # Content-Based Filtering Demo
    print("\n🎯 Content-Based Recommendations (Similar to 'The Matrix'):")
    content_recs = rec_system.content_based_recommendations(1)  # The Matrix
    rec_system.display_recommendations(content_recs, "Content-Based Filtering")
    
    # Hybrid Recommendations Demo
    print("\n🔄 Hybrid Recommendations for User 1:")
    hybrid_recs = rec_system.hybrid_recommendations(1)
    rec_system.display_recommendations(hybrid_recs, "Hybrid Recommendations")
    
    # Add new rating demo
    print("\n➕ Adding New Rating Demo:")
    rec_system.add_user_rating(1, 3, 5)  # User 1 rates Titanic with 5 stars
    
    print("\n🔄 Updated Hybrid Recommendations for User 1:")
    updated_recs = rec_system.hybrid_recommendations(1)
    rec_system.display_recommendations(updated_recs, "Updated Hybrid Recommendations")


if __name__ == "__main__":
    main()