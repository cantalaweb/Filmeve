#!/usr/bin/env python3
"""
Data Cleaning Script

Usage:
    python src/clean_data.py
"""
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_data(data_dir: Path = Path('data/processed')):
    """Clean enriched ratings data."""
    logger.info("="*60)
    logger.info("DATA CLEANING")
    logger.info("="*60)

    # Load enriched data
    logger.info("\nLoading enriched data...")
    df = pd.read_csv(data_dir / 'ratings_enriched.csv')
    logger.info(f"✓ Loaded {len(df):,} rows")

    # Check missing data
    logger.info("\nChecking missing data...")
    missing_director = df['director'].isna().sum()
    missing_cast = df['cast'].isna().sum()
    logger.info(f"  Missing director: {missing_director:,}")
    logger.info(f"  Missing cast: {missing_cast:,}")

    # Remove ID columns (keep userId and movieId - essential for recommendations)
    logger.info("\nRemoving unnecessary ID columns...")
    id_columns = ['imdbId', 'tmdbId']
    columns_to_drop = [col for col in id_columns if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        logger.info(f"✓ Dropped: {', '.join(columns_to_drop)}")

    # Remove non-essential columns
    logger.info("\nRemoving non-essential columns...")
    other_columns = ['timestamp', 'tagline', 'overview']
    columns_to_drop = [col for col in other_columns if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        logger.info(f"✓ Dropped: {', '.join(columns_to_drop)}")

    # Remove rows with missing critical data
    logger.info("\nRemoving rows with missing critical data...")
    rows_before = len(df)

    critical_columns = [
        'director', 'cast', 'country', 'runtime', 'budget',
        'revenue', 'release_date', 'original_language',
        'vote_average', 'vote_count', 'popularity'
    ]

    df = df.dropna(subset=critical_columns).copy()
    rows_after = len(df)
    removed = rows_before - rows_after

    logger.info(f"✓ Removed {removed:,} rows with missing data")
    logger.info(f"✓ Remaining: {rows_after:,} rows")

    # Clean title (remove year suffixes)
    logger.info("\nCleaning titles...")
    df['title'] = df['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    logger.info("✓ Removed year suffixes from titles")

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("CLEANING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total ratings: {len(df):,}")
    logger.info(f"Unique users: {df['userId'].nunique():,}")
    logger.info(f"Unique movies: {df['movieId'].nunique():,}")
    logger.info(f"Unique directors: {df['director'].nunique():,}")
    logger.info(f"Columns: {len(df.columns)}")

    logger.info(f"\nRating statistics:")
    logger.info(f"  Mean: {df['rating'].mean():.2f}")
    logger.info(f"  Median: {df['rating'].median():.1f}")
    logger.info(f"  Std: {df['rating'].std():.2f}")

    # Save cleaned data
    logger.info("\nSaving cleaned data...")
    output_path = data_dir / 'ratings_cleaned.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved to: {output_path}")

    logger.info("\n" + "="*60)
    logger.info("CLEANING COMPLETE")
    logger.info("="*60)


def main():
    try:
        clean_data()
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Run data sourcing first: python src/source_data.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Cleaning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
