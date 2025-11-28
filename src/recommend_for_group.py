#!/usr/bin/env python3
"""
Group Movie Recommendation System - Production Script

Usage:
    # Production: Recommend movies for a group of users
    python src/recommend_for_group.py --users 105,275,357,202,62

    # Test mode: Evaluate on ALL discovered test groups
    python src/recommend_for_group.py --test

    # Test mode: Evaluate on a SPECIFIC test group
    python src/recommend_for_group.py --test "Steven Spielberg Fans"

    # Custom parameters
    python src/recommend_for_group.py --users 1,2,3,4,5 --slate-size 10 --candidates 200

Note: Test mode requires running discover_users_diverse.py first to generate test groups.
"""
import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import joblib
import pickle
from tqdm import tqdm

# Suppress sklearn feature name warnings (harmless for production)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GroupRecommender:
    """Production-ready group movie recommender."""

    def __init__(self, model_dir: Path = Path('models'), data_dir: Path = Path('data/processed')):
        """Initialize recommender by loading bias-reduced model and data."""
        logger.info("Loading bias-reduced model and data...")

        # Load bias-reduced model (always use the best performing model)
        self.model = joblib.load(model_dir / 'model_bias_reduced.pkl')
        self.feature_columns = joblib.load(model_dir / 'feature_columns_bias_reduced.pkl')

        # Load bias-reduced data
        self.df_ratings = pd.read_csv(data_dir / 'ratings_bias_reduced.csv')
        self.df_features = pd.read_csv(data_dir / 'ratings_featured_bias_reduced.csv')
        self.df_movies = pd.read_csv(data_dir / 'movies_enriched.csv')

        logger.info(f"✓ Bias-reduced model loaded ({len(self.feature_columns)} features)")
        logger.info(f"✓ Data loaded: {len(self.df_ratings):,} ratings, {len(self.df_movies):,} movies")
        logger.info(f"✓ Featured data loaded: {len(self.df_features):,} samples")

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a user-movie pair."""
        # Check for exact match
        existing = self.df_features[
            (self.df_features['userId'] == user_id) &
            (self.df_features['movieId'] == movie_id)
        ]

        if len(existing) > 0:
            X = existing[self.feature_columns].values
            return float(self.model.predict(X)[0])

        # Create synthetic features for unseen pair
        user_features = self.df_features[self.df_features['userId'] == user_id]
        movie_features = self.df_features[self.df_features['movieId'] == movie_id]

        if len(user_features) == 0 or len(movie_features) == 0:
            return 3.5  # Default

        user_sample = user_features.iloc[0]
        movie_avg = movie_features[self.feature_columns].mean()

        # Mix user and movie features
        synthetic_features = []
        for col in self.feature_columns:
            if col.startswith('user_'):
                synthetic_features.append(user_sample[col])
            elif col.startswith('movie_') or col.startswith('genre_'):
                synthetic_features.append(movie_avg[col])
            else:
                synthetic_features.append((user_sample[col] + movie_avg[col]) / 2)

        X = np.array(synthetic_features).reshape(1, -1)
        return float(self.model.predict(X)[0])

    def generate_candidates(self, user_ids: List[int], n_candidates: int = 200) -> pd.DataFrame:
        """Generate candidate movies for the group."""
        logger.info(f"Generating candidates for {len(user_ids)} users...")

        all_movie_ids = set(self.df_movies['movieId'].unique())
        candidate_predictions = defaultdict(list)

        for user_id in user_ids:
            # Find unseen movies
            rated_movies = set(self.df_ratings[self.df_ratings['userId'] == user_id]['movieId'].unique())
            unseen_movies = list(all_movie_ids - rated_movies)

            if len(unseen_movies) == 0:
                logger.warning(f"User {user_id} has rated all movies!")
                continue

            # Sample for speed
            sample_size = min(500, len(unseen_movies))
            sampled = np.random.choice(unseen_movies, sample_size, replace=False)

            # Predict ratings
            for movie_id in sampled:
                pred = self.predict_rating(user_id, movie_id)
                candidate_predictions[movie_id].append(pred)

        # Score by average prediction
        candidate_scores = [(mid, np.mean(preds)) for mid, preds in candidate_predictions.items()]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        # Create DataFrame
        top_candidates = candidate_scores[:n_candidates]
        candidates_df = self.df_movies[self.df_movies['movieId'].isin([m for m, _ in top_candidates])].copy()
        score_dict = {m: s for m, s in top_candidates}
        candidates_df['group_score'] = candidates_df['movieId'].map(score_dict)
        candidates_df = candidates_df.sort_values('group_score', ascending=False)

        logger.info(f"✓ Generated {len(candidates_df)} candidates")
        return candidates_df

    def optimize_slate(
        self,
        user_ids: List[int],
        candidates_df: pd.DataFrame,
        slate_size: int = 10,
        population_size: int = 60,
        generations: int = 35
    ) -> Dict[str, Any]:
        """Optimize movie slate using genetic algorithm."""
        logger.info("Running genetic algorithm optimization...")

        candidate_movie_ids = candidates_df['movieId'].tolist()

        # GA helper functions
        def calculate_fitness(slate):
            user_scores = []
            for user_id in user_ids:
                user_ratings = [self.predict_rating(user_id, mid) for mid in slate]
                user_scores.append(np.mean(user_ratings))

            avg_satisfaction = np.mean(user_scores)
            disagreement = np.std(user_scores)
            fitness = 0.8 * avg_satisfaction - 0.2 * disagreement

            return fitness, avg_satisfaction, disagreement, user_scores

        def create_initial_population():
            return [
                np.random.choice(candidate_movie_ids, size=slate_size, replace=False).tolist()
                for _ in range(population_size)
            ]

        def tournament_selection(population, fitnesses, tournament_size=3):
            idx = np.random.choice(len(population), tournament_size, replace=False)
            winner_idx = idx[np.argmax([fitnesses[i] for i in idx])]
            return population[winner_idx]

        def crossover(parent1, parent2):
            size = len(parent1)
            p1, p2 = sorted(np.random.choice(size, 2, replace=False))

            child1 = [None] * size
            child2 = [None] * size
            child1[p1:p2] = parent1[p1:p2]
            child2[p1:p2] = parent2[p1:p2]

            def fill(child, parent, p1, p2):
                child_set = set(child[p1:p2])
                remaining = [g for g in parent if g not in child_set]
                idx = 0
                for pos in list(range(0, p1)) + list(range(p2, size)):
                    child[pos] = remaining[idx]
                    idx += 1
                return child

            return fill(child1, parent2, p1, p2), fill(child2, parent1, p1, p2)

        def mutate(slate, mutation_rate=0.05):
            """Replacement mutation - swap one movie with a candidate from pool."""
            if np.random.random() < mutation_rate:
                # Pick random position to mutate
                idx = np.random.choice(len(slate))

                # Pick random candidate NOT currently in slate
                available_candidates = [m for m in candidate_movie_ids if m not in slate]

                if len(available_candidates) > 0:
                    new_movie = np.random.choice(available_candidates)
                    slate[idx] = new_movie  # Replace with new movie from pool

            return slate

        # Run GA
        population = create_initial_population()
        elite_size = 10

        best_fitness_history = []

        for generation in tqdm(range(generations), desc='Optimizing'):
            # Evaluate
            results = [calculate_fitness(slate) for slate in population]
            fitnesses = [r[0] for r in results]

            # Track best
            best_idx = np.argmax(fitnesses)
            best_fitness_history.append(fitnesses[best_idx])

            # Elitism
            elite_idx = np.argsort(fitnesses)[-elite_size:]
            new_population = [population[i] for i in elite_idx]

            # Breed
            while len(new_population) < population_size:
                p1 = tournament_selection(population, fitnesses)
                p2 = tournament_selection(population, fitnesses)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                new_population.extend([c1, c2])

            population = new_population[:population_size]

        # Final evaluation
        final_results = [calculate_fitness(slate) for slate in population]
        final_fitnesses = [r[0] for r in final_results]
        best_idx = np.argmax(final_fitnesses)

        logger.info("✓ Optimization complete")

        return {
            'best_slate': population[best_idx],
            'fitness': final_results[best_idx][0],
            'satisfaction': final_results[best_idx][1],
            'disagreement': final_results[best_idx][2],
            'user_scores': final_results[best_idx][3],
            'best_fitness_history': best_fitness_history
        }

    def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Get profile for a single user."""
        user_ratings = self.df_ratings[self.df_ratings['userId'] == user_id]

        if len(user_ratings) == 0:
            return None

        # Basic stats
        profile = {
            'userId': user_id,
            'rating_count': len(user_ratings),
            'avg_rating': float(user_ratings['rating'].mean()),
            'std_rating': float(user_ratings['rating'].std()),
            'min_rating': float(user_ratings['rating'].min()),
            'max_rating': float(user_ratings['rating'].max()),
            'median_rating': float(user_ratings['rating'].median())
        }

        # Genre distribution
        genre_counts = defaultdict(int)
        for genres_str in user_ratings['genres'].dropna():
            for genre in genres_str.split('|'):
                genre_counts[genre.strip()] += 1

        # Top genres (sorted by count)
        profile['top_genres'] = sorted(
            genre_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 genres

        return profile

    def recommend(
        self,
        user_ids: List[int],
        slate_size: int = 10,
        n_candidates: int = 200,
        generations: int = 35
    ) -> Dict[str, Any]:
        """
        Generate movie recommendations for a group of users.

        Args:
            user_ids: List of user IDs in the group
            slate_size: Number of movies to recommend
            n_candidates: Size of candidate pool
            generations: Number of GA generations

        Returns:
            Dictionary with recommended slate and metrics
        """
        # Validate users
        valid_users = []
        for uid in user_ids:
            if uid in self.df_ratings['userId'].values:
                valid_users.append(uid)
            else:
                logger.warning(f"User {uid} not found in dataset - skipping")

        if len(valid_users) == 0:
            raise ValueError("No valid users provided")

        logger.info(f"Recommending for group: {valid_users}")

        # Get user profiles
        profiles = []
        for uid in valid_users:
            profile = self.get_user_profile(uid)
            if profile:
                profiles.append(profile)

        # Generate candidates
        candidates_df = self.generate_candidates(valid_users, n_candidates)

        # Optimize slate
        result = self.optimize_slate(valid_users, candidates_df, slate_size, generations=generations)

        # Add movie details and user profiles
        slate_movies = self.df_movies[self.df_movies['movieId'].isin(result['best_slate'])]
        result['movies'] = slate_movies.to_dict('records')
        result['user_ids'] = valid_users
        result['user_profiles'] = profiles

        return result


def format_results(result: Dict[str, Any], show_profiles: bool = True) -> str:
    """Format results for display."""
    output = []
    output.append("=" * 80)
    output.append("GROUP MOVIE RECOMMENDATIONS")
    output.append("=" * 80)
    output.append(f"\nGroup: {len(result['user_ids'])} users")
    output.append(f"Satisfaction: {result['satisfaction']:.2f} / 5.0")
    output.append(f"Disagreement: {result['disagreement']:.2f}")
    output.append(f"Fitness: {result['fitness']:.3f}")

    # User profiles (if available and requested)
    if show_profiles and 'user_profiles' in result and result['user_profiles']:
        output.append(f"\n{'='*80}")
        output.append("USER PROFILES:")
        output.append("=" * 80)

        for profile in result['user_profiles']:
            output.append(f"\n👤 User {profile['userId']}")
            output.append(f"   Ratings: {profile['rating_count']} movies")
            output.append(f"   Average: {profile['avg_rating']:.2f} (±{profile['std_rating']:.2f})")
            output.append(f"   Range:   {profile['min_rating']:.1f} - {profile['max_rating']:.1f} (median: {profile['median_rating']:.1f})")

            # Characterize user
            if profile['avg_rating'] >= 4.0:
                user_type = "Generous rater"
            elif profile['avg_rating'] <= 3.0:
                user_type = "Critical rater"
            else:
                user_type = "Balanced rater"

            if profile['std_rating'] < 0.8:
                consistency = "Consistent"
            elif profile['std_rating'] > 1.2:
                consistency = "Varied"
            else:
                consistency = "Moderate"

            output.append(f"   Type:    {user_type}, {consistency} ratings")

            # Top genres
            if profile['top_genres']:
                genres_str = ", ".join([f"{g} ({c})" for g, c in profile['top_genres']])
                output.append(f"   Genres:  {genres_str}")

    output.append(f"\n{'='*80}")
    output.append("RECOMMENDED SLATE:")
    output.append("=" * 80)

    for i, movie_id in enumerate(result['best_slate'], 1):
        movie = next(m for m in result['movies'] if m['movieId'] == movie_id)
        output.append(f"{i:2d}. {movie['title']}")
        output.append(f"    Genres: {movie['genres']}")

    output.append(f"\n{'='*80}")
    output.append("INDIVIDUAL USER SCORES:")
    output.append("=" * 80)

    for user_id, score in zip(result['user_ids'], result['user_scores']):
        output.append(f"  User {user_id:3d}: {score:.2f}")

    # Genre distribution
    genre_counts = defaultdict(int)
    for movie in result['movies']:
        if pd.notna(movie['genres']):
            for genre in movie['genres'].split('|'):
                genre_counts[genre.strip()] += 1

    output.append(f"\n{'='*80}")
    output.append("GENRE DISTRIBUTION:")
    output.append("=" * 80)

    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(result['best_slate']) * 100
        output.append(f"  {genre:20s} {count:2d} ({pct:5.1f}%)")

    return "\n".join(output)


def test_mode(recommender: GroupRecommender, group_name: str = 'all'):
    """
    Run in test mode - evaluate on discovered test groups.

    Args:
        recommender: Initialized GroupRecommender
        group_name: Name of specific group to test, or 'all' for all groups
    """
    logger.info("Running in TEST mode")

    # Load test groups
    test_groups_path = Path('data/processed/test_groups_diverse.json')
    if not test_groups_path.exists():
        logger.error("Test groups not found.")
        logger.error("Please run user discovery first: python src/discover_users_diverse.py")
        sys.exit(1)

    with open(test_groups_path) as f:
        test_groups = json.load(f)

    logger.info(f"Found {len(test_groups)} test groups")

    # Filter groups if specific group requested
    if group_name != 'all':
        test_groups = [g for g in test_groups if g['name'] == group_name]
        if not test_groups:
            available = [g['name'] for g in json.load(open(test_groups_path))]
            logger.error(f"Group '{group_name}' not found.")
            logger.error(f"Available groups: {', '.join(available)}")
            sys.exit(1)
        logger.info(f"Testing specific group: {group_name}")
    else:
        logger.info(f"Testing all {len(test_groups)} groups")

    # Test each group
    all_results = []
    for group in test_groups:
        logger.info(f"\nTesting: {group['name']}")
        logger.info(f"Users: {group['users']}")

        result = recommender.recommend(group['users'])
        result['group_name'] = group['name']
        all_results.append(result)

        # Display results (without profiles in test mode)
        print("\n" + format_results(result, show_profiles=False))

    # Summary (only if testing multiple groups)
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        for result in all_results:
            print(f"{result['group_name']:30s} | Satisfaction: {result['satisfaction']:.2f} | Disagreement: {result['disagreement']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate group movie recommendations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--users',
        type=str,
        help='Comma-separated list of user IDs (e.g., "105,275,357,202,62")'
    )

    parser.add_argument(
        '--test',
        nargs='?',
        const='all',
        metavar='GROUP_NAME',
        help='Run in test mode. Specify group name or leave empty for all groups (e.g., --test "Steven Spielberg Fans" or --test)'
    )

    parser.add_argument(
        '--slate-size',
        type=int,
        default=10,
        help='Number of movies to recommend (default: 10)'
    )

    parser.add_argument(
        '--candidates',
        type=int,
        default=200,
        help='Size of candidate pool (default: 200)'
    )

    parser.add_argument(
        '--generations',
        type=int,
        default=35,
        help='Number of GA generations (default: 35)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.test is None and not args.users:
        parser.error("Either --users or --test must be specified")

    # Initialize recommender
    try:
        recommender = GroupRecommender()
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {e}")
        sys.exit(1)

    # Run
    if args.test is not None:
        test_mode(recommender, group_name=args.test)
    else:
        # Parse user IDs
        try:
            user_ids = [int(uid.strip()) for uid in args.users.split(',')]
        except ValueError:
            logger.error("Invalid user IDs. Must be comma-separated integers.")
            sys.exit(1)

        # Generate recommendations
        try:
            result = recommender.recommend(
                user_ids,
                slate_size=args.slate_size,
                n_candidates=args.candidates,
                generations=args.generations
            )

            # Display
            print("\n" + format_results(result))

            # Save if requested
            if args.output:
                # Convert numpy types to Python types for JSON
                result_copy = result.copy()
                result_copy['user_scores'] = [float(s) for s in result_copy['user_scores']]
                result_copy['best_fitness_history'] = [float(f) for f in result_copy['best_fitness_history']]

                with open(args.output, 'w') as f:
                    json.dump(result_copy, f, indent=2)
                logger.info(f"Results saved to {args.output}")

        except Exception as e:
            logger.error(f"Recommendation failed: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
