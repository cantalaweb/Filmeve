#!/usr/bin/env python3
"""
Feature Engineering Module

Complete feature engineering pipeline extracted from notebook 03.
Ensures identical feature set (100 features) for fair comparison.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def engineer_features(df: pd.DataFrame, global_mean: float) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete feature engineering pipeline from notebook 03.

    Args:
        df: DataFrame with [userId, movieId, rating, title, genres, director, cast, country,
                           runtime, budget, revenue, release_date, original_language,
                           vote_average, vote_count, popularity, tags]
        global_mean: Global mean rating (for encoding)

    Returns:
        X: Features (100 columns)
        y: Target (rating)
    """
    # 1. USER FEATURES
    user_stats = df.groupby('userId').agg(
        user_rating_count=('rating', 'count'),
        user_avg_rating=('rating', 'mean'),
        user_rating_std=('rating', 'std'),
        user_rating_min=('rating', 'min'),
        user_rating_max=('rating', 'max'),
        user_rating_median=('rating', 'median')
    ).reset_index()
    user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)
    user_stats['user_rating_range'] = user_stats['user_rating_max'] - user_stats['user_rating_min']

    # User genre preferences
    df_exploded = df.copy()
    df_exploded['genre_list'] = df_exploded['genres'].str.split('|')
    df_exploded = df_exploded.explode('genre_list')
    # Filter out NaN values before sorting
    df_exploded = df_exploded[df_exploded['genre_list'].notna()]
    all_genres = sorted(df_exploded['genre_list'].unique())

    user_genre_prefs = df_exploded.groupby(['userId', 'genre_list'])['rating'].mean().unstack(fill_value=0)
    user_genre_prefs.columns = [f'user_pref_{g.lower().replace("-", "_")}' for g in user_genre_prefs.columns]
    user_genre_prefs = user_genre_prefs.reset_index()

    user_genre_diversity = df_exploded.groupby('userId')['genre_list'].nunique().reset_index()
    user_genre_diversity.columns = ['userId', 'user_genre_diversity']

    user_director_diversity = df.groupby('userId')['director'].nunique().reset_index()
    user_director_diversity.columns = ['userId', 'user_director_diversity']

    user_country_diversity = df.groupby('userId')['country'].nunique().reset_index()
    user_country_diversity.columns = ['userId', 'user_country_diversity']

    user_features = user_stats.merge(user_genre_prefs, on='userId', how='left')
    user_features = user_features.merge(user_genre_diversity, on='userId', how='left')
    user_features = user_features.merge(user_director_diversity, on='userId', how='left')
    user_features = user_features.merge(user_country_diversity, on='userId', how='left')

    # 2. MOVIE FEATURES
    movie_features = df.groupby('movieId').agg(
        title=('title', 'first'),
        runtime=('runtime', 'first'),
        budget=('budget', 'first'),
        revenue=('revenue', 'first'),
        vote_average=('vote_average', 'first'),
        vote_count=('vote_count', 'first'),
        popularity=('popularity', 'first'),
        release_date=('release_date', 'first'),
        original_language=('original_language', 'first'),
        country=('country', 'first'),
        director=('director', 'first'),
        cast=('cast', 'first'),
        genres=('genres', 'first'),
        movie_rating_count=('rating', 'count'),
        movie_avg_rating=('rating', 'mean'),
        movie_rating_std=('rating', 'std')
    ).reset_index()
    movie_features['movie_rating_std'] = movie_features['movie_rating_std'].fillna(0)

    # Temporal features
    movie_features['release_date'] = pd.to_datetime(movie_features['release_date'], errors='coerce')
    movie_features['release_year'] = movie_features['release_date'].dt.year
    movie_features['release_month'] = movie_features['release_date'].dt.month
    movie_features['release_day_of_week'] = movie_features['release_date'].dt.dayofweek
    movie_features['movie_age'] = 2025 - movie_features['release_year']
    movie_features['is_recent'] = (movie_features['release_year'] >= 2015).astype(int)
    movie_features['decade'] = (movie_features['release_year'] // 10) * 10

    # Financial features
    movie_features['budget_log'] = np.log1p(movie_features['budget'])
    movie_features['revenue_log'] = np.log1p(movie_features['revenue'])
    movie_features['roi'] = np.where(
        movie_features['budget'] > 0,
        (movie_features['revenue'] - movie_features['budget']) / movie_features['budget'],
        0
    )
    movie_features['profit'] = movie_features['revenue'] - movie_features['budget']
    movie_features['profit_log'] = np.sign(movie_features['profit']) * np.log1p(np.abs(movie_features['profit']))
    movie_features['is_blockbuster'] = (movie_features['revenue'] > 100_000_000).astype(int)
    movie_features['is_high_budget'] = (movie_features['budget'] > 50_000_000).astype(int)

    # Popularity features
    movie_features['popularity_log'] = np.log1p(movie_features['popularity'])
    movie_features['vote_count_log'] = np.log1p(movie_features['vote_count'])
    movie_features['engagement_score'] = (
        movie_features['popularity_log'] * 0.5 +
        movie_features['vote_count_log'] * 0.5
    )

    # Genre encoding
    for genre in all_genres:
        col_name = f'genre_{genre.lower().replace("-", "_")}'
        movie_features[col_name] = movie_features['genres'].str.contains(genre, regex=False, na=False).astype(int)
    movie_features['genre_count'] = movie_features['genres'].fillna('').str.count('\\|') + 1
    movie_features.loc[movie_features['genres'].isna(), 'genre_count'] = 0

    # Director encoding
    director_stats = df.groupby('director').agg(
        director_avg_rating=('rating', 'mean'),
        director_rating_count=('rating', 'count'),
        director_movie_count=('movieId', 'nunique')
    ).reset_index()

    min_samples = 10
    director_stats['director_encoded'] = (
        (director_stats['director_avg_rating'] * director_stats['director_rating_count'] +
         global_mean * min_samples) /
        (director_stats['director_rating_count'] + min_samples)
    )

    movie_features = movie_features.merge(
        director_stats[['director', 'director_encoded', 'director_rating_count', 'director_movie_count']],
        on='director', how='left'
    )

    # Cast encoding
    df_cast = df[['movieId', 'cast', 'rating']].copy()
    df_cast['actor_list'] = df_cast['cast'].str.split('|')
    df_cast = df_cast.explode('actor_list')
    # Filter out NaN values
    df_cast = df_cast[df_cast['actor_list'].notna()]

    actor_stats = df_cast.groupby('actor_list').agg(
        actor_avg_rating=('rating', 'mean'),
        actor_rating_count=('rating', 'count')
    ).reset_index()

    movie_features['lead_actor'] = movie_features['cast'].str.split('|').str[0]
    movie_features['second_actor'] = movie_features['cast'].str.split('|').str[1]

    lead_actor_stats = actor_stats.rename(columns={
        'actor_list': 'lead_actor',
        'actor_avg_rating': 'lead_actor_avg_rating',
        'actor_rating_count': 'lead_actor_rating_count'
    })
    movie_features = movie_features.merge(lead_actor_stats, on='lead_actor', how='left')

    second_actor_stats = actor_stats.rename(columns={
        'actor_list': 'second_actor',
        'actor_avg_rating': 'second_actor_avg_rating',
        'actor_rating_count': 'second_actor_rating_count'
    })
    movie_features = movie_features.merge(second_actor_stats, on='second_actor', how='left')

    def get_cast_avg_rating(cast_str):
        if pd.isna(cast_str):
            return global_mean
        actors = cast_str.split('|')
        ratings = []
        for actor in actors:
            actor_row = actor_stats[actor_stats['actor_list'] == actor]
            if len(actor_row) > 0:
                ratings.append(actor_row['actor_avg_rating'].values[0])
        return np.mean(ratings) if ratings else global_mean

    movie_features['cast_avg_rating'] = movie_features['cast'].apply(get_cast_avg_rating)

    movie_features['lead_actor_avg_rating'] = movie_features['lead_actor_avg_rating'].fillna(global_mean)
    movie_features['lead_actor_rating_count'] = movie_features['lead_actor_rating_count'].fillna(0)
    movie_features['second_actor_avg_rating'] = movie_features['second_actor_avg_rating'].fillna(global_mean)
    movie_features['second_actor_rating_count'] = movie_features['second_actor_rating_count'].fillna(0)

    # Language/Country encoding
    language_stats = df.groupby('original_language')['rating'].mean().reset_index()
    language_stats.columns = ['original_language', 'language_avg_rating']
    movie_features = movie_features.merge(language_stats, on='original_language', how='left')
    movie_features['language_avg_rating'] = movie_features['language_avg_rating'].fillna(global_mean)
    movie_features['is_spanish'] = (movie_features['original_language'] == 'es').astype(int)
    movie_features['is_english'] = (movie_features['original_language'] == 'en').astype(int)

    country_stats = df.groupby('country')['rating'].mean().reset_index()
    country_stats.columns = ['country', 'country_avg_rating']
    movie_features = movie_features.merge(country_stats, on='country', how='left')
    movie_features['country_avg_rating'] = movie_features['country_avg_rating'].fillna(global_mean)
    movie_features['is_spain'] = (movie_features['country'] == 'ES').astype(int)
    movie_features['is_us'] = (movie_features['country'] == 'US').astype(int)

    # Quality features
    movie_features['tmdb_scaled'] = movie_features['vote_average'] / 2
    movie_features['rating_vs_tmdb'] = movie_features['movie_avg_rating'] - movie_features['tmdb_scaled']
    movie_features['is_high_tmdb'] = (movie_features['vote_average'] >= 7.0).astype(int)
    movie_features['is_well_reviewed'] = (
        (movie_features['vote_average'] >= 7.0) &
        (movie_features['vote_count'] >= 1000)
    ).astype(int)

    # 3. MERGE & INTERACTION - This creates duplicate _x and _y columns!
    df_features = df.merge(user_features, on='userId', how='left')
    df_features = df_features.merge(
        movie_features.drop(columns=['director', 'cast', 'genres', 'release_date', 'original_language', 'country', 'lead_actor']),
        on='movieId', how='left'
    )

    # User-genre match score
    genre_cols = [col for col in movie_features.columns if col.startswith('genre_')]
    user_pref_cols = [col for col in user_features.columns if col.startswith('user_pref_')]

    genre_to_pref = {}
    for genre in all_genres:
        genre_col = f'genre_{genre.lower().replace("-", "_")}'
        pref_col = f'user_pref_{genre.lower().replace("-", "_")}'
        if genre_col in genre_cols and pref_col in user_pref_cols:
            genre_to_pref[genre_col] = pref_col

    match_sum = np.zeros(len(df_features))
    genre_count = np.zeros(len(df_features))

    for genre_col, pref_col in genre_to_pref.items():
        if genre_col in df_features.columns and pref_col in df_features.columns:
            mask = df_features[genre_col] == 1
            match_sum += mask * df_features[pref_col]
            genre_count += mask

    df_features['user_genre_match'] = np.where(genre_count > 0, match_sum / genre_count, global_mean)

    # Deviation features
    df_features['rating_vs_user_avg'] = df_features['movie_avg_rating'] - df_features['user_avg_rating']
    df_features['tmdb_vs_user_avg'] = df_features['tmdb_scaled'] - df_features['user_avg_rating']
    df_features['director_vs_user_avg'] = df_features['director_encoded'] - df_features['user_avg_rating']

    # Relative features
    avg_user_ratings = user_stats['user_rating_count'].mean()
    df_features['user_activity_ratio'] = df_features['user_rating_count'] / avg_user_ratings

    avg_movie_ratings = movie_features['movie_rating_count'].mean()
    df_features['movie_popularity_ratio'] = df_features['movie_rating_count'] / avg_movie_ratings

    df_features['user_leniency'] = df_features['user_avg_rating'] - global_mean

    # 4. PREPARE FEATURES
    exclude_cols = [
        'userId', 'title', 'genres', 'director', 'cast', 'country',
        'original_language', 'release_date', 'tags', 'lead_actor', 'second_actor',
        'rating',  # target
        'budget', 'revenue', 'profit',  # use log versions
        'popularity', 'vote_count',  # use log versions
        'runtime_x', 'runtime_y',  # duplicates
        'tmdb_scaled'  # intermediate
    ]

    feature_cols = [col for col in df_features.columns
                    if col not in exclude_cols and df_features[col].dtype in ['int64', 'float64']]

    X = df_features[feature_cols].copy()
    y = df_features['rating'].copy()
    X = X.fillna(X.median())

    return X, y
