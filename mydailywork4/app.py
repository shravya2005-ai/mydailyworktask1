from flask import Flask, render_template, request, jsonify, redirect, url_for
from recommendation_system import RecommendationSystem
import json

app = Flask(__name__)

# Initialize the recommendation system
rec_system = RecommendationSystem()
rec_system.load_sample_data()
rec_system.build_content_similarity()

@app.route('/')
def index():
    """Main page showing all movies and users"""
    movies = rec_system.movies_df.to_dict('records')
    users = rec_system.ratings_df['user_id'].unique().tolist()
    return render_template('index.html', movies=movies, users=users)

@app.route('/user/<int:user_id>')
def user_profile(user_id):
    """User profile page with recommendations"""
    profile = rec_system.get_user_profile(user_id)
    
    if isinstance(profile, str):  # Error message
        return render_template('error.html', message=profile)
    
    # Get all three types of recommendations
    collab_recs = rec_system.collaborative_filtering_recommendations(user_id, 5)
    hybrid_recs = rec_system.hybrid_recommendations(user_id, 5)
    
    # Get content-based recommendations for user's highest rated movie
    content_recs = []
    if profile['rated_movies']:
        highest_rated = max(profile['rated_movies'], key=lambda x: x['rating'])
        movie_id = rec_system.movies_df[rec_system.movies_df['title'] == highest_rated['title']]['movie_id'].iloc[0]
        content_recs = rec_system.content_based_recommendations(movie_id, 5)
    
    return render_template('user_profile.html', 
                         profile=profile, 
                         collab_recs=collab_recs,
                         content_recs=content_recs,
                         hybrid_recs=hybrid_recs)

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    """Movie details page with similar movies"""
    movie = rec_system.movies_df[rec_system.movies_df['movie_id'] == movie_id]
    
    if movie.empty:
        return render_template('error.html', message=f"Movie {movie_id} not found")
    
    movie_info = movie.iloc[0].to_dict()
    similar_movies = rec_system.content_based_recommendations(movie_id, 5)
    
    # Get ratings for this movie
    movie_ratings = rec_system.ratings_df[rec_system.ratings_df['movie_id'] == movie_id]
    avg_rating = movie_ratings['rating'].mean() if not movie_ratings.empty else 0
    
    return render_template('movie_details.html', 
                         movie=movie_info, 
                         similar_movies=similar_movies,
                         avg_rating=round(avg_rating, 1),
                         total_ratings=len(movie_ratings))

@app.route('/add_rating', methods=['POST'])
def add_rating():
    """Add a new rating via AJAX"""
    try:
        user_id = int(request.form['user_id'])
        movie_id = int(request.form['movie_id'])
        rating = int(request.form['rating'])
        
        rec_system.add_user_rating(user_id, movie_id, rating)
        
        return jsonify({
            'success': True, 
            'message': f'Rating added successfully! User {user_id} rated movie {movie_id} with {rating} stars.'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/recommendations/<int:user_id>')
def api_recommendations(user_id):
    """API endpoint for getting recommendations"""
    rec_type = request.args.get('type', 'hybrid')
    
    try:
        if rec_type == 'collaborative':
            recs = rec_system.collaborative_filtering_recommendations(user_id)
        elif rec_type == 'content':
            # Get content-based recs for user's highest rated movie
            profile = rec_system.get_user_profile(user_id)
            if isinstance(profile, dict) and profile['rated_movies']:
                highest_rated = max(profile['rated_movies'], key=lambda x: x['rating'])
                movie_id = int(rec_system.movies_df[rec_system.movies_df['title'] == highest_rated['title']]['movie_id'].iloc[0])
                recs = rec_system.content_based_recommendations(movie_id)
            else:
                recs = []
        else:  # hybrid
            recs = rec_system.hybrid_recommendations(user_id)
        
        # Ensure all values are JSON serializable
        if isinstance(recs, list):
            for rec in recs:
                if isinstance(rec, dict):
                    for key, value in rec.items():
                        if hasattr(value, 'item'):  # numpy scalar
                            rec[key] = value.item()
                        elif hasattr(value, 'tolist'):  # numpy array
                            rec[key] = value.tolist()
        
        return jsonify(recs)
    except Exception as e:
        print(f"Error in api_recommendations: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/compare')
def compare_recommendations():
    """Page to compare different recommendation methods"""
    users = rec_system.ratings_df['user_id'].unique().tolist()
    return render_template('compare.html', users=users)

if __name__ == '__main__':
    app.run(debug=True, port=5000)