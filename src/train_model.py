#!/usr/bin/env python3
"""
Model Training Script - Bias-Reduced Recommendation Model

Usage:
    # Train model with 5 winning bias reduction strategies
    python src/train_model.py

    # Test saved bias-reduced model
    python src/train_model.py --test

    # Custom parameters
    python src/train_model.py --cv-folds 5 --test-size 0.2

The 5 winning bias reduction strategies (always applied):
    1. expectation: Correct expectation-driven rating inflation (+3.09% improvement)
    2. cinephile: Adjust for cinephile vs casual viewer differences (+0.83%)
    3. collaborative: Collaborative debiasing based on similar users (+0.68%)
    4. popularity: Debias popularity effects (+0.35%)
    5. review_bombing: Detect and downweight review bombing (+0.26%)

Combined improvement: +4.31% over baseline (RMSE: 0.6932 vs 0.7244)
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Import custom modules
from bias_reduction import BiasReducer
from feature_engineering import engineer_features

# 5 winning bias reduction strategies (from notebook 04 evaluation)
BEST_STRATEGIES = ['expectation', 'cinephile', 'collaborative', 'popularity', 'review_bombing']

# Suppress warnings for clean output
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Bias-reduced model training pipeline."""

    def __init__(self, data_dir: Path = Path('data/processed')):
        """Initialize trainer.

        Args:
            data_dir: Directory containing processed data
        """
        self.data_dir = data_dir
        self.ratings = None
        self.movies = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.global_mean = None

    def load_data(self):
        """Load enriched ratings and movies data."""
        logger.info("Loading enriched data...")

        # Load ratings with TMDB metadata
        self.ratings = pd.read_csv(self.data_dir / 'ratings_enriched.csv')
        logger.info(f"✓ Loaded {len(self.ratings):,} ratings with TMDB metadata")

        # Load movies for bias reduction
        self.movies = pd.read_csv(self.data_dir / 'movies_enriched.csv')
        logger.info(f"✓ Loaded {len(self.movies):,} movies")

        # Calculate global mean for feature engineering
        self.global_mean = self.ratings['rating'].mean()
        logger.info(f"✓ Global mean rating: {self.global_mean:.3f}")

    def apply_bias_reduction(self):
        """Apply the 5 winning bias reduction strategies to ratings."""
        logger.info(f"Applying 5 winning bias reduction strategies:")
        for i, strategy in enumerate(BEST_STRATEGIES, 1):
            logger.info(f"  {i}. {strategy}")

        # Extract core columns for BiasReducer
        ratings_core = self.ratings[['userId', 'movieId', 'rating', 'timestamp']].copy()

        # Apply bias reduction
        reducer = BiasReducer(ratings_core, self.movies)
        ratings_reduced = reducer.apply_strategies(BEST_STRATEGIES)

        # Merge bias-reduced ratings back with metadata
        self.ratings = self.ratings.drop(columns=['rating']).merge(
            ratings_reduced[['userId', 'movieId', 'rating']],
            on=['userId', 'movieId'],
            how='inner'
        )

        logger.info(f"✓ Bias reduction complete: {len(self.ratings):,} ratings")

    def prepare_features(self):
        """Apply complete feature engineering pipeline."""
        logger.info("Applying complete feature engineering pipeline...")

        # Use the feature engineering module
        self.X, self.y = engineer_features(self.ratings, self.global_mean)

        logger.info(f"✓ Features: {self.X.shape[1]}")
        logger.info(f"✓ Samples: {len(self.X):,}")

    def split_data(self, test_size: float = 0.2):
        """Split data into train and test sets."""
        logger.info(f"Splitting data (test_size={test_size})...")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )

        logger.info(f"✓ Train: {len(self.X_train):,} samples")
        logger.info(f"✓ Test:  {len(self.X_test):,} samples")

    def train_model(self, cv_folds: int = 3):
        """Train stacked ensemble model."""
        logger.info("Training model...")

        # Define base estimators
        estimators = [
            ('xgb', xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=1.0,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            )),
            ('lgb', lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )),
            ('cat', CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=0
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ))
        ]

        # Create shuffled KFold for reproducible CV regardless of row order
        cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Create stacking regressor
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=cv_splitter,
            n_jobs=-1,
            passthrough=False
        )

        logger.info(f"  Training with {cv_folds}-fold shuffled CV (random_state=42)...")
        model.fit(self.X_train, self.y_train)

        logger.info("✓ Model training complete")
        return model

    def evaluate_model(self, model):
        """Evaluate model performance on train and test sets."""
        logger.info("Evaluating model...")

        # Predictions on pre-split data
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # Metrics
        metrics = {
            'train_rmse': float(np.sqrt(mean_squared_error(self.y_train, y_train_pred))),
            'test_rmse': float(np.sqrt(mean_squared_error(self.y_test, y_test_pred))),
            'train_mae': float(mean_absolute_error(self.y_train, y_train_pred)),
            'test_mae': float(mean_absolute_error(self.y_test, y_test_pred)),
            'train_r2': float(r2_score(self.y_train, y_train_pred)),
            'test_r2': float(r2_score(self.y_test, y_test_pred))
        }

        logger.info(f"✓ Evaluation complete")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"  Test MAE:  {metrics['test_mae']:.4f}")
        logger.info(f"  Test R²:   {metrics['test_r2']:.4f}")

        return metrics

    def save_model(self, model, metrics, model_dir: Path = Path('models')):
        """Save bias-reduced model and metadata."""
        logger.info("Saving bias-reduced model...")

        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model (always bias-reduced)
        model_path = model_dir / 'model_bias_reduced.pkl'
        joblib.dump(model, model_path)

        # Save feature columns (needed for inference)
        feature_cols_path = model_dir / 'feature_columns_bias_reduced.pkl'
        joblib.dump(list(self.X.columns), feature_cols_path)

        # Save config
        config = {
            'model_type': 'Stacked Ensemble (Bias-Reduced)',
            'base_models': ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting'],
            'meta_learner': 'Ridge',
            'bias_reduction': {
                'enabled': True,
                'strategies': BEST_STRATEGIES,
                'num_strategies': len(BEST_STRATEGIES)
            },
            'metrics': metrics,
            'n_features': self.X.shape[1],
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'timestamp': datetime.now().isoformat()
        }

        config_path = model_dir / 'model_bias_reduced_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"✓ Saved model to {model_path}")
        logger.info(f"✓ Saved config to {config_path}")

        # Also save the bias-reduced ratings for future use
        ratings_path = self.data_dir / 'ratings_bias_reduced.csv'
        self.ratings.to_csv(ratings_path, index=False)
        logger.info(f"✓ Saved bias-reduced ratings to {ratings_path}")

        # Save featured data for production inference
        # Combine features with userId, movieId, and rating for lookup
        df_featured = self.X.copy()
        df_featured['userId'] = self.ratings['userId'].values
        df_featured['movieId'] = self.ratings['movieId'].values
        df_featured['rating'] = self.y.values
        df_featured['title'] = self.ratings['title'].values

        featured_path = self.data_dir / 'ratings_featured_bias_reduced.csv'
        df_featured.to_csv(featured_path, index=False)
        logger.info(f"✓ Saved featured data to {featured_path}")


def test_saved_model(test_size: float = 0.2):
    """Test the saved bias-reduced model."""
    logger.info("=" * 60)
    logger.info("TESTING SAVED BIAS-REDUCED MODEL")
    logger.info("=" * 60)

    model_dir = Path('models')
    data_dir = Path('data/processed')

    # Load saved model
    logger.info("Loading saved model...")
    model = joblib.load(model_dir / 'model_bias_reduced.pkl')
    feature_columns = joblib.load(model_dir / 'feature_columns_bias_reduced.pkl')
    logger.info(f"✓ Loaded model with {len(feature_columns)} features")

    # Load bias-reduced ratings
    logger.info("Loading bias-reduced ratings...")
    ratings = pd.read_csv(data_dir / 'ratings_bias_reduced.csv')
    logger.info(f"✓ Loaded {len(ratings):,} bias-reduced ratings")

    # Feature engineering
    global_mean = ratings['rating'].mean()
    logger.info("Applying feature engineering...")
    X, y = engineer_features(ratings, global_mean)
    logger.info(f"✓ Generated {X.shape[1]} features")

    # Verify feature alignment
    if list(X.columns) != feature_columns:
        logger.warning("Feature columns don't match! Reordering...")
        X = X[feature_columns]

    # Split BEFORE evaluation (same as training)
    logger.info(f"Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    logger.info(f"✓ Train: {len(X_train):,} samples")
    logger.info(f"✓ Test:  {len(X_test):,} samples")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Train RMSE: {train_rmse:.4f}")
    logger.info(f"Test RMSE:  {test_rmse:.4f}")
    logger.info(f"Train MAE:  {train_mae:.4f}")
    logger.info(f"Test MAE:   {test_mae:.4f}")
    logger.info(f"Train R²:   {train_r2:.4f}")
    logger.info(f"Test R²:    {test_r2:.4f}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Train or test bias-reduced recommendation model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train model with 5 winning bias reduction strategies
    python src/train_model.py

    # Test saved bias-reduced model
    python src/train_model.py --test
        """
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test the saved bias-reduced model instead of training'
    )

    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size for evaluation (default: 0.2)'
    )

    args = parser.parse_args()

    # Test mode - evaluate saved model
    if args.test:
        try:
            test_saved_model(test_size=args.test_size)
            sys.exit(0)
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Training mode - train new model with 5 strategies
    trainer = ModelTrainer()

    try:
        # Pipeline
        trainer.load_data()
        trainer.apply_bias_reduction()  # Apply 5 winning strategies
        trainer.prepare_features()
        trainer.split_data(test_size=args.test_size)

        # Train (on training set only)
        model = trainer.train_model(cv_folds=args.cv_folds)

        # Evaluate
        metrics = trainer.evaluate_model(model)

        # Save
        trainer.save_model(model, metrics)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model: Stacked Ensemble (Bias-Reduced)")
        logger.info(f"Bias Reduction Strategies: {', '.join(BEST_STRATEGIES)}")
        logger.info(f"Features: {trainer.X.shape[1]}")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"Test MAE:  {metrics['test_mae']:.4f}")
        logger.info(f"Test R²:   {metrics['test_r2']:.4f}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
