#!/usr/bin/env python3
"""
Bias Reduction Module - Apply bias reduction strategies to ratings data.

This module implements all bias reduction strategies from planning/sesgos.md:
1. Implicit Negative Preferences - Add synthetic low ratings for unwatched genres
2. Binary Voter Normalization - Normalize users who rate in binary mode
3. Z-Score Normalization - Normalize skewed rating distributions
4. Temporal Weighting - Weight recent ratings more heavily
5. Genre Deviation - Adjust for user genre preferences
6. Popularity Debiasing - Downweight overly popular movies
7. Expectation Correction - Adjust for hype effects
8. Review Bombing Detection - Detect and handle review bombing
9. Cinephile Adjustment - Separate casual vs cinephile users
10. Collaborative Debiasing - Peer-based normalization

Usage:
    from bias_reduction import BiasReducer

    reducer = BiasReducer(ratings_df, movies_df)
    ratings_reduced = reducer.apply_strategies(['zscore', 'binary', 'temporal'])
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


class BiasReducer:
    """Apply bias reduction strategies to ratings data."""

    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Initialize bias reducer.

        Args:
            ratings_df: DataFrame with userId, movieId, rating, timestamp
            movies_df: DataFrame with movieId, title, genres
        """
        self.ratings_original = ratings_df.copy()
        self.movies = movies_df.copy()
        self.global_mean = ratings_df['rating'].mean()
        self.global_std = ratings_df['rating'].std()

        # Strategy mapping
        self.strategies = {
            'implicit_negatives': self._implicit_negatives,
            'binary': self._binary_normalization,
            'zscore': self._zscore_normalization,
            'temporal': self._temporal_weighting,
            'genre_deviation': self._genre_deviation,
            'popularity': self._popularity_debiasing,
            'expectation': self._expectation_correction,
            'review_bombing': self._review_bombing_detection,
            'cinephile': self._cinephile_adjustment,
            'collaborative': self._collaborative_debiasing
        }

    def apply_strategies(self, strategy_names: List[str]) -> pd.DataFrame:
        """
        Apply selected bias reduction strategies sequentially.

        Args:
            strategy_names: List of strategy names to apply

        Returns:
            DataFrame with bias-reduced ratings
        """
        ratings = self.ratings_original.copy()

        for strategy_name in strategy_names:
            if strategy_name not in self.strategies:
                logger.warning(f"Unknown strategy: {strategy_name}")
                continue

            logger.info(f"Applying strategy: {strategy_name}")
            ratings = self.strategies[strategy_name](ratings)

        return ratings

    # ========================================================================
    # STRATEGY 1: IMPLICIT NEGATIVE PREFERENCES
    # ========================================================================

    def _get_all_genres(self) -> Set[str]:
        """Extract all unique genres."""
        all_genres = set()
        for genres_str in self.movies['genres'].dropna():
            if genres_str != '(no genres listed)':
                all_genres.update(genres_str.split('|'))
        return all_genres

    def _implicit_negatives(
        self,
        ratings_df: pd.DataFrame,
        min_user_ratings: int = 20,
        min_genre_movies: int = 10,
        implicit_rating: float = 0.4,
        movies_per_genre: int = 3
    ) -> pd.DataFrame:
        """Add synthetic low ratings for unwatched genres."""
        all_genres = self._get_all_genres()

        # Filter genres with sufficient representation
        valid_genres = set()
        for genre in all_genres:
            count = self.movies['genres'].str.contains(genre, na=False).sum()
            if count >= min_genre_movies:
                valid_genres.add(genre)

        # Merge with genres
        ratings_with_genres = ratings_df.merge(
            self.movies[['movieId', 'genres']], on='movieId'
        )

        # Active users
        user_counts = ratings_df.groupby('userId').size()
        active_users = user_counts[user_counts >= min_user_ratings].index

        # Generate synthetic ratings
        synthetic_ratings = []

        for user_id in active_users:
            user_data = ratings_with_genres[ratings_with_genres['userId'] == user_id]

            # Get watched genres
            watched_genres = set()
            for genres_str in user_data['genres'].dropna():
                if genres_str != '(no genres listed)':
                    watched_genres.update(genres_str.split('|'))

            unwatched_genres = valid_genres - watched_genres
            median_timestamp = user_data['timestamp'].median()

            # Add synthetic ratings for unwatched genres
            for genre in unwatched_genres:
                genre_movies = self.movies[
                    self.movies['genres'].str.contains(genre, na=False)
                ]
                sample_size = min(movies_per_genre, len(genre_movies))

                if sample_size > 0:
                    sample_movies = genre_movies.sample(sample_size, random_state=42)

                    for movie_id in sample_movies['movieId']:
                        if movie_id not in user_data['movieId'].values:
                            synthetic_ratings.append({
                                'userId': user_id,
                                'movieId': movie_id,
                                'rating': implicit_rating,
                                'timestamp': median_timestamp,
                                'is_synthetic': True
                            })

        # Combine
        ratings_enhanced = ratings_df.copy()
        ratings_enhanced['is_synthetic'] = False

        if synthetic_ratings:
            synthetic_df = pd.DataFrame(synthetic_ratings)
            ratings_enhanced = pd.concat([ratings_enhanced, synthetic_df], ignore_index=True)

        logger.info(f"  Added {len(synthetic_ratings):,} synthetic negatives")
        return ratings_enhanced

    # ========================================================================
    # STRATEGY 2: BINARY VOTER NORMALIZATION
    # ========================================================================

    def _binary_normalization(
        self,
        ratings_df: pd.DataFrame,
        min_ratings: int = 20,
        extreme_threshold: float = 0.7
    ) -> pd.DataFrame:
        """Detect and normalize binary voters."""
        ratings_normalized = ratings_df.copy()
        binary_voters = []

        for user_id, user_ratings in ratings_df.groupby('userId'):
            if len(user_ratings) < min_ratings:
                continue

            ratings_array = user_ratings['rating'].values

            # Count extremes
            low_extremes = ((ratings_array >= 0.5) & (ratings_array <= 2.0)).sum()
            high_extremes = ((ratings_array >= 4.0) & (ratings_array <= 5.0)).sum()
            extreme_fraction = (low_extremes + high_extremes) / len(ratings_array)

            middle_fraction = ((ratings_array > 2.0) & (ratings_array < 4.0)).sum() / len(ratings_array)

            if extreme_fraction >= extreme_threshold and middle_fraction < 0.3:
                binary_voters.append(user_id)

                # Min-max normalize to [0.5, 5.0]
                user_mask = ratings_normalized['userId'] == user_id
                user_ratings_vals = ratings_normalized.loc[user_mask, 'rating'].values

                min_rating = user_ratings_vals.min()
                max_rating = user_ratings_vals.max()

                if max_rating > min_rating:
                    normalized = 0.5 + (user_ratings_vals - min_rating) / (max_rating - min_rating) * 4.5
                    ratings_normalized.loc[user_mask, 'rating'] = normalized

        logger.info(f"  Normalized {len(binary_voters)} binary voters")
        return ratings_normalized

    # ========================================================================
    # STRATEGY 3: Z-SCORE NORMALIZATION
    # ========================================================================

    def _zscore_normalization(
        self,
        ratings_df: pd.DataFrame,
        min_ratings: int = 20,
        global_mean: float = 3.5,
        global_std: float = 1.0
    ) -> pd.DataFrame:
        """Apply z-score normalization per user."""
        ratings_zscore = ratings_df.copy()
        normalized_count = 0

        for user_id, user_ratings in ratings_df.groupby('userId'):
            if len(user_ratings) < min_ratings:
                continue

            user_mask = ratings_zscore['userId'] == user_id
            user_ratings_vals = ratings_zscore.loc[user_mask, 'rating'].values

            user_mean = user_ratings_vals.mean()
            user_std = user_ratings_vals.std()

            if user_std > 0.1:
                z_scores = (user_ratings_vals - user_mean) / user_std
                normalized = global_mean + z_scores * global_std
                normalized = np.clip(normalized, 0.5, 5.0)

                ratings_zscore.loc[user_mask, 'rating'] = normalized
                normalized_count += 1

        logger.info(f"  Normalized {normalized_count} users")
        return ratings_zscore

    # ========================================================================
    # STRATEGY 4: TEMPORAL WEIGHTING
    # ========================================================================

    def _temporal_weighting(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal weights to favor recent ratings."""
        ratings_temporal = ratings_df.copy()
        ratings_temporal = ratings_temporal.sort_values(['userId', 'timestamp'])

        # Calculate position in user's rating history
        ratings_temporal['user_rating_sequence'] = ratings_temporal.groupby('userId').cumcount() + 1
        ratings_temporal['user_total_ratings'] = ratings_temporal.groupby('userId')['userId'].transform('count')
        ratings_temporal['user_rating_percentile'] = (
            (ratings_temporal['user_rating_sequence'] - 1) /
            (ratings_temporal['user_total_ratings'] - 1)
        ).fillna(0.5)

        # Weight: 0.5 (first) to 1.0 (latest)
        ratings_temporal['temporal_weight'] = 0.5 + 0.5 * ratings_temporal['user_rating_percentile']

        # Drop temporary columns
        ratings_temporal = ratings_temporal.drop(
            columns=['user_rating_sequence', 'user_total_ratings', 'user_rating_percentile']
        )

        logger.info("  Added temporal weights")
        return ratings_temporal

    # ========================================================================
    # STRATEGY 5: GENRE DEVIATION
    # ========================================================================

    def _genre_deviation(
        self,
        ratings_df: pd.DataFrame,
        min_genre_ratings: int = 3
    ) -> pd.DataFrame:
        """Calculate deviation from user's typical genre rating."""
        # This strategy is complex and adds a feature column rather than modifying ratings
        # For production, we skip this as it's better handled in feature engineering
        logger.info("  Genre deviation (skipped - use in feature engineering)")
        return ratings_df

    # ========================================================================
    # STRATEGY 6: POPULARITY DEBIASING
    # ========================================================================

    def _popularity_debiasing(
        self,
        ratings_df: pd.DataFrame,
        adjustment_factor: float = 0.1
    ) -> pd.DataFrame:
        """Adjust ratings for overly popular movies."""
        ratings_adjusted = ratings_df.copy()

        # Calculate movie popularity
        movie_popularity = ratings_df.groupby('movieId').size()
        popularity_percentile = movie_popularity.rank(pct=True)

        # Create popularity tier
        popularity_map = popularity_percentile.to_dict()
        ratings_adjusted['popularity_percentile'] = ratings_adjusted['movieId'].map(popularity_map)

        # Adjust very popular movies (top 10%)
        blockbuster_mask = ratings_adjusted['popularity_percentile'] > 0.9
        adjustment = adjustment_factor * (ratings_adjusted['popularity_percentile'] - 0.9) * 10
        ratings_adjusted.loc[blockbuster_mask, 'rating'] -= adjustment[blockbuster_mask]
        ratings_adjusted['rating'] = ratings_adjusted['rating'].clip(0.5, 5.0)

        # Drop temporary column
        ratings_adjusted = ratings_adjusted.drop(columns=['popularity_percentile'])

        logger.info(f"  Adjusted {blockbuster_mask.sum()} blockbuster ratings")
        return ratings_adjusted

    # ========================================================================
    # STRATEGY 7: EXPECTATION CORRECTION
    # ========================================================================

    def _expectation_correction(
        self,
        ratings_df: pd.DataFrame,
        franchise_penalty: float = 0.15
    ) -> pd.DataFrame:
        """Adjust for expectation effects (sequels, franchises)."""
        ratings_adjusted = ratings_df.copy()
        ratings_adjusted = ratings_adjusted.merge(
            self.movies[['movieId', 'title']], on='movieId'
        )

        # Detect sequels/franchises
        sequel_keywords = ['II', 'III', 'Part', '2', '3', '4', '5', 'Reloaded', 'Revolutions', 'Returns']
        is_sequel = ratings_adjusted['title'].str.contains(
            '|'.join(sequel_keywords), case=False, na=False
        )

        # Detect early ratings
        ratings_adjusted = ratings_adjusted.sort_values(['movieId', 'timestamp'])
        ratings_adjusted['movie_rating_sequence'] = ratings_adjusted.groupby('movieId').cumcount() + 1
        ratings_adjusted['movie_total_ratings'] = ratings_adjusted.groupby('movieId')['movieId'].transform('count')
        is_early_rating = (
            ratings_adjusted['movie_rating_sequence'] / ratings_adjusted['movie_total_ratings']
        ) <= 0.2

        # Apply correction
        correction_mask = is_sequel & is_early_rating
        ratings_adjusted.loc[correction_mask, 'rating'] = (
            ratings_adjusted.loc[correction_mask, 'rating'] * (1 - franchise_penalty) +
            self.global_mean * franchise_penalty
        )

        # Cleanup
        ratings_adjusted = ratings_adjusted.drop(
            columns=['title', 'movie_rating_sequence', 'movie_total_ratings']
        )

        logger.info(f"  Corrected {correction_mask.sum()} early sequel ratings")
        return ratings_adjusted

    # ========================================================================
    # STRATEGY 8: REVIEW BOMBING DETECTION
    # ========================================================================

    def _review_bombing_detection(
        self,
        ratings_df: pd.DataFrame,
        spike_threshold: float = 3.0
    ) -> pd.DataFrame:
        """Detect and downweight potential review bombing."""
        ratings_adjusted = ratings_df.copy()
        ratings_adjusted = ratings_adjusted.sort_values(['movieId', 'timestamp'])

        # Convert timestamp to days since first rating
        ratings_adjusted['days_since_first'] = ratings_adjusted.groupby('movieId')['timestamp'].transform(
            lambda x: (x - x.min()) / (24 * 3600)
        )

        # Bin into weeks
        ratings_adjusted['week_bin'] = (ratings_adjusted['days_since_first'] // 7).astype(int)

        # Calculate weekly rating rate
        weekly_rates = ratings_adjusted.groupby(['movieId', 'week_bin']).size().reset_index(name='weekly_count')
        avg_weekly_rate = weekly_rates.groupby('movieId')['weekly_count'].mean().reset_index(name='avg_weekly_count')

        # Merge back
        ratings_adjusted = ratings_adjusted.merge(
            weekly_rates[['movieId', 'week_bin', 'weekly_count']],
            on=['movieId', 'week_bin'],
            how='left'
        )
        ratings_adjusted = ratings_adjusted.merge(avg_weekly_rate, on='movieId', how='left')

        # Detect spikes
        is_spike = ratings_adjusted['weekly_count'] > (spike_threshold * ratings_adjusted['avg_weekly_count'])
        is_extreme = (ratings_adjusted['rating'] <= 1.5) | (ratings_adjusted['rating'] >= 4.5)
        is_bombing = is_spike & is_extreme

        # Downweight
        ratings_adjusted.loc[is_bombing, 'rating'] = (
            ratings_adjusted.loc[is_bombing, 'rating'] * 0.7 + self.global_mean * 0.3
        )

        # Cleanup
        ratings_adjusted = ratings_adjusted.drop(
            columns=['days_since_first', 'week_bin', 'weekly_count', 'avg_weekly_count']
        )

        logger.info(f"  Adjusted {is_bombing.sum()} suspected review bombing ratings")
        return ratings_adjusted

    # ========================================================================
    # STRATEGY 9: CINEPHILE ADJUSTMENT
    # ========================================================================

    def _cinephile_adjustment(
        self,
        ratings_df: pd.DataFrame,
        n_clusters: int = 3
    ) -> pd.DataFrame:
        """Detect cinephile users and adjust their ratings."""
        # Calculate user sophistication metrics
        user_stats = ratings_df.groupby('userId').agg(
            n_ratings=('rating', 'count'),
            avg_rating=('rating', 'mean'),
            std_rating=('rating', 'std'),
        ).reset_index()

        # Count unique genres per user
        ratings_with_genres = ratings_df.merge(
            self.movies[['movieId', 'genres']], on='movieId'
        )
        user_genre_diversity = []

        for user_id in user_stats['userId']:
            user_genres = ratings_with_genres[ratings_with_genres['userId'] == user_id]['genres']
            all_genres = set()
            for genres_str in user_genres.dropna():
                if genres_str != '(no genres listed)':
                    all_genres.update(genres_str.split('|'))
            user_genre_diversity.append(len(all_genres))

        user_stats['genre_diversity'] = user_genre_diversity

        # Cluster users
        features = user_stats[['n_ratings', 'std_rating', 'genre_diversity']].fillna(0)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        user_stats['user_cluster'] = kmeans.fit_predict(features_scaled)

        # Identify cinephile cluster
        cluster_profiles = user_stats.groupby('user_cluster').agg({
            'n_ratings': 'mean',
            'std_rating': 'mean',
            'genre_diversity': 'mean'
        })

        cinephile_cluster = cluster_profiles['genre_diversity'].idxmax()
        user_stats['is_cinephile'] = (user_stats['user_cluster'] == cinephile_cluster).astype(int)

        # Merge back
        ratings_adjusted = ratings_df.merge(user_stats[['userId', 'is_cinephile']], on='userId')

        # Adjust cinephile ratings
        cinephile_mask = ratings_adjusted['is_cinephile'] == 1
        ratings_adjusted.loc[cinephile_mask, 'rating'] += 0.2
        ratings_adjusted['rating'] = ratings_adjusted['rating'].clip(0.5, 5.0)

        # Cleanup
        ratings_adjusted = ratings_adjusted.drop(columns=['is_cinephile'])

        logger.info(f"  Identified {cinephile_mask.sum()} cinephile ratings")
        return ratings_adjusted

    # ========================================================================
    # STRATEGY 10: COLLABORATIVE DEBIASING
    # ========================================================================

    def _collaborative_debiasing(
        self,
        ratings_df: pd.DataFrame,
        n_similar_users: int = 10,
        adjustment_strength: float = 0.2
    ) -> pd.DataFrame:
        """Adjust ratings based on similar users' calibration."""
        # Create user-movie matrix
        user_movie_matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )

        # Calculate user similarity (sample for efficiency)
        if len(user_movie_matrix) > 500:
            sample_users = user_movie_matrix.sample(500, random_state=42)
            user_corr = sample_users.T.corr()
        else:
            user_corr = user_movie_matrix.T.corr()

        # Find similar users and calculate adjustments
        ratings_adjusted = ratings_df.copy()
        user_adjustments = {}

        for user_id in user_corr.index:
            similar_users = user_corr[user_id].drop(user_id).nlargest(n_similar_users)

            if len(similar_users) == 0:
                user_adjustments[user_id] = 0
                continue

            # Calculate rating bias difference
            user_mean = ratings_df[ratings_df['userId'] == user_id]['rating'].mean()
            similar_means = []

            for similar_id in similar_users.index:
                similar_mean = ratings_df[ratings_df['userId'] == similar_id]['rating'].mean()
                similar_means.append(similar_mean)

            avg_similar_mean = np.mean(similar_means)
            bias_difference = user_mean - avg_similar_mean

            user_adjustments[user_id] = -bias_difference * adjustment_strength

        # Apply adjustments
        ratings_adjusted['adjustment'] = ratings_adjusted['userId'].map(user_adjustments).fillna(0)
        ratings_adjusted['rating'] += ratings_adjusted['adjustment']
        ratings_adjusted['rating'] = ratings_adjusted['rating'].clip(0.5, 5.0)

        # Cleanup
        ratings_adjusted = ratings_adjusted.drop(columns=['adjustment'])

        n_adjusted = (ratings_adjusted['userId'].map(user_adjustments).fillna(0).abs() > 0.01).sum()
        logger.info(f"  Adjusted {n_adjusted} ratings via peer normalization")
        return ratings_adjusted


def apply_bias_reduction(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    strategies: List[str] = None
) -> pd.DataFrame:
    """
    Convenience function to apply bias reduction strategies.

    Args:
        ratings_df: Ratings DataFrame
        movies_df: Movies DataFrame
        strategies: List of strategy names (default: ['zscore', 'binary', 'temporal'])

    Returns:
        Bias-reduced ratings DataFrame
    """
    if strategies is None:
        strategies = ['zscore', 'binary', 'temporal']

    reducer = BiasReducer(ratings_df, movies_df)
    return reducer.apply_strategies(strategies)
