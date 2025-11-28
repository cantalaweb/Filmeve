"""
Utility functions for Filmeve Streamlit app
"""
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import os
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables
load_dotenv(dotenv_path='../.env')
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

@lru_cache(maxsize=100)
def get_poster_url(title, year=None):
    """Get poster URL from TMDB (cached)"""
    if not TMDB_API_KEY:
        return None

    try:
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': title,
        }
        if year:
            params['year'] = year

        response = requests.get(search_url, params=params, timeout=5)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                poster_path = results[0].get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w300{poster_path}"
    except:
        pass
    return None

def get_tmdb_poster(title, year=None, tmdb_id=None):
    """
    Fetch movie poster from TMDB (with caching)
    Returns PIL Image or None
    """
    poster_url = get_poster_url(title, year)

    if poster_url:
        try:
            img_response = requests.get(poster_url, timeout=5)
            if img_response.status_code == 200:
                return Image.open(BytesIO(img_response.content))
        except:
            pass

    # Fallback: create placeholder
    try:
        img = Image.new('RGB', (300, 450), color=(102, 126, 234))
        return img
    except:
        return None

def calculate_compatibility_matrix(user_ids, user_data):
    """
    Calculate taste compatibility between users
    Based on correlation of their ratings
    """
    compatibility = np.zeros((len(user_ids), len(user_ids)))

    for i, user1 in enumerate(user_ids):
        for j, user2 in enumerate(user_ids):
            if i == j:
                compatibility[i][j] = 1.0
            else:
                # Get common movies
                user1_ratings = user_data[user_data['userId'] == user1]
                user2_ratings = user_data[user_data['userId'] == user2]

                common_movies = set(user1_ratings['movieId']) & set(user2_ratings['movieId'])

                if len(common_movies) > 5:
                    u1_common = user1_ratings[user1_ratings['movieId'].isin(common_movies)].sort_values('movieId')['rating'].values
                    u2_common = user2_ratings[user2_ratings['movieId'].isin(common_movies)].sort_values('movieId')['rating'].values

                    # Pearson correlation
                    corr = np.corrcoef(u1_common, u2_common)[0, 1]
                    compatibility[i][j] = max(0, corr)  # Clip negative correlations to 0
                else:
                    compatibility[i][j] = 0.5  # Neutral if not enough data

    return compatibility

def create_compatibility_heatmap(user_ids, user_names, compatibility_matrix):
    """
    Create a beautiful heatmap showing taste compatibility
    """
    labels = [f"{user_names[uid][0]}" for uid in user_ids]

    fig = go.Figure(data=go.Heatmap(
        z=compatibility_matrix,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=np.round(compatibility_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Compatibility")
    ))

    fig.update_layout(
        title="Taste Compatibility Matrix",
        xaxis_title="",
        yaxis_title="",
        height=400,
        width=600
    )

    return fig

def explain_movie_choice(movie, user_ratings, user_names):
    """
    Generate explanation for why this movie works for the group
    """
    avg_rating = np.mean(list(user_ratings.values()))
    std_rating = np.std(list(user_ratings.values()))

    explanations = []

    # High satisfaction
    if avg_rating >= 4.0:
        explanations.append("**Strong consensus** - Everyone loves this one")
    elif avg_rating >= 3.5:
        explanations.append("**Solid choice** - Good ratings across the board")
    else:
        explanations.append("**Balanced pick** - Compromises well for different tastes")

    # Low disagreement
    if std_rating < 0.3:
        explanations.append("**High agreement** - Very similar expectations")
    elif std_rating < 0.5:
        explanations.append("**Moderate agreement** - Most people aligned")
    else:
        explanations.append("**Diverse appeal** - Something for everyone")

    # Individual callouts
    top_fan = max(user_ratings.items(), key=lambda x: x[1])
    if top_fan[1] >= 4.5:
        name = user_names[top_fan[0]][0]
        explanations.append(f"**{name}'s favorite** - Rated {top_fan[1]:.1f}/5.0")

    return " • ".join(explanations)

def get_genre_distribution(movies_data, movie_ids):
    """
    Get genre distribution for a set of movies
    """
    genres = {}

    for movie_id in movie_ids:
        try:
            movie = movies_data[movies_data['movieId'] == movie_id]
            if len(movie) > 0:
                # Try different possible column names
                genre_col = None
                if 'genres' in movie.columns:
                    genre_col = 'genres'
                elif 'genre' in movie.columns:
                    genre_col = 'genre'

                if genre_col:
                    movie_genres = str(movie.iloc[0][genre_col]).split('|')
                    for genre in movie_genres:
                        genre = genre.strip()
                        if genre and genre.lower() != 'nan' and genre != '(no genres listed)':
                            genres[genre] = genres.get(genre, 0) + 1
        except Exception as e:
            continue

    return genres

def create_genre_pie_chart(genre_dist):
    """
    Create pie chart of genre distribution
    """
    if not genre_dist:
        return None

    labels = list(genre_dist.keys())
    values = list(genre_dist.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker_colors=px.colors.qualitative.Set3
    )])

    fig.update_layout(
        title="Recommended Genres",
        height=300
    )

    return fig

def calculate_group_stats(user_ids, user_data, user_names, recommendations):
    """
    Calculate interesting statistics about the group
    """
    stats = {}

    # Most compatible pair
    compatibility = calculate_compatibility_matrix(user_ids, user_data)
    max_compat = 0
    best_pair = None

    for i in range(len(user_ids)):
        for j in range(i+1, len(user_ids)):
            if compatibility[i][j] > max_compat:
                max_compat = compatibility[i][j]
                best_pair = (user_ids[i], user_ids[j])

    if best_pair:
        stats['most_compatible'] = (
            f"{user_names[best_pair[0]][0]} & "
            f"{user_names[best_pair[1]][0]}",
            max_compat
        )

    # Biggest critic (lowest average predicted rating)
    avg_ratings = {}
    for user_id in user_ids:
        ratings = [m['user_ratings'].get(user_id, 3.0) for m in recommendations]
        avg_ratings[user_id] = np.mean(ratings)

    critic = min(avg_ratings.items(), key=lambda x: x[1])
    stats['biggest_critic'] = (
        f"{user_names[critic[0]][0]}",
        critic[1]
    )

    # Most enthusiastic (highest average predicted rating)
    enthusiast = max(avg_ratings.items(), key=lambda x: x[1])
    stats['most_enthusiastic'] = (
        f"{user_names[enthusiast[0]][0]}",
        enthusiast[1]
    )

    # Hardest to please (highest disagreement contribution)
    disagreements = {}
    for user_id in user_ids:
        user_ratings = [m['user_ratings'].get(user_id, 3.0) for m in recommendations]
        group_avgs = [m['avg_group_rating'] for m in recommendations]
        disagreements[user_id] = np.mean([abs(ur - ga) for ur, ga in zip(user_ratings, group_avgs)])

    hardest = max(disagreements.items(), key=lambda x: x[1])
    stats['hardest_to_please'] = (
        f"{user_names[hardest[0]][0]}",
        hardest[1]
    )

    return stats

def create_pdf_export(recommendations, user_ids, user_names):
    """
    Create a simple text export of recommendations
    Returns string for download
    """
    export_text = "FILMEVE - YOUR PERFECT MOVIE SLATE\n"
    export_text += "=" * 50 + "\n\n"

    export_text += "YOUR MOVIE CREW:\n"
    for user_id in user_ids:
        first_name, nickname = user_names[user_id]
        export_text += f"  {first_name}, {nickname}\n"

    export_text += "\n" + "=" * 50 + "\n\n"
    export_text += "TOP RECOMMENDATIONS:\n\n"

    for idx, movie in enumerate(recommendations[:10], 1):
        export_text += f"{idx}. {movie['title']}\n"
        export_text += f"   Group Rating: {movie['avg_group_rating']:.2f}/5.0\n"
        export_text += f"   Disagreement: {movie['disagreement']:.2f}\n"
        export_text += "   Individual Ratings:\n"
        for user_id, rating in movie['user_ratings'].items():
            first_name = user_names[user_id][0]
            export_text += f"     • {first_name}: {rating:.2f}/5.0\n"
        export_text += "\n"

    export_text += "=" * 50 + "\n"
    export_text += "Powered by Filmeve AI\n"

    return export_text

def compare_individual_vs_group(user_ids, user_data, model, feature_cols, group_recommendations):
    """
    Compare what each person would pick individually vs group consensus
    """
    individual_picks = {}

    for user_id in user_ids:
        # Get user's data
        user_movies = user_data[user_data['userId'] == user_id].copy()

        # Predict ratings for all their movies
        predictions = []
        for idx, row in user_movies.iterrows():
            # Pass as DataFrame to preserve feature names
            features = user_movies.loc[[idx], feature_cols]
            pred = model.predict(features)[0]
            predictions.append({
                'movieId': row['movieId'],
                'title': row['title'],
                'predicted_rating': pred
            })

        # Sort by predicted rating
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        individual_picks[user_id] = predictions[:5]  # Top 5

    return individual_picks
