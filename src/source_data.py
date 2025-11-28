#!/usr/bin/env python3
"""
Data Sourcing Script - Load MovieLens and enrich with TMDB data

Usage:
    python src/source_data.py

    # Or if you have already scraped TMDB data
    python src/source_data.py --skip-scraping
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def source_data(data_dir: Path = Path('data'), skip_scraping: bool = False):
    """Load and enrich MovieLens data with TMDB."""
    logger.info("="*60)
    logger.info("DATA SOURCING")
    logger.info("="*60)

    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'

    # Load MovieLens ratings
    logger.info("\nLoading MovieLens ratings...")
    ratings_path = raw_dir / 'ratings.csv'
    if not ratings_path.exists():
        logger.error(f"Ratings file not found: {ratings_path}")
        logger.error("Please download MovieLens dataset and place in data/raw/")
        sys.exit(1)

    df_ratings = pd.read_csv(ratings_path)
    logger.info(f"✓ Loaded {len(df_ratings):,} ratings")
    logger.info(f"  Users: {df_ratings['userId'].nunique():,}")
    logger.info(f"  Movies: {df_ratings['movieId'].nunique():,}")

    # Load MovieLens movies
    logger.info("\nLoading MovieLens movies...")
    movies_path = raw_dir / 'movies.csv'
    if not movies_path.exists():
        logger.error(f"Movies file not found: {movies_path}")
        sys.exit(1)

    df_movies = pd.read_csv(movies_path)
    logger.info(f"✓ Loaded {len(df_movies):,} movies")

    # Load MovieLens links (for TMDB IDs)
    logger.info("\nLoading MovieLens links...")
    links_path = raw_dir / 'links.csv'
    if not links_path.exists():
        logger.error(f"Links file not found: {links_path}")
        sys.exit(1)

    df_links = pd.read_csv(links_path)
    logger.info(f"✓ Loaded {len(df_links):,} links")

    # Load tags (optional)
    tags_path = raw_dir / 'tags.csv'
    if tags_path.exists():
        logger.info("\nLoading MovieLens tags...")
        df_tags = pd.read_csv(tags_path)

        # Aggregate tags per user-movie
        df_tags_agg = df_tags.groupby(['userId', 'movieId'])['tag'].apply(lambda x: '|'.join(x)).reset_index()
        df_tags_agg.columns = ['userId', 'movieId', 'tags']

        logger.info(f"✓ Loaded {len(df_tags):,} tags")
    else:
        df_tags_agg = None
        logger.info("\nNo tags file found - skipping")

    # Merge MovieLens data
    logger.info("\nMerging MovieLens data...")
    df = df_ratings.merge(df_movies, on='movieId', how='left')
    df = df.merge(df_links, on='movieId', how='left')

    if df_tags_agg is not None:
        df = df.merge(df_tags_agg, on=['userId', 'movieId'], how='left')

    logger.info(f"✓ Merged: {len(df):,} rows")

    # Check for TMDB data
    tmdb_path = processed_dir / 'movies_enriched.csv'

    if not tmdb_path.exists():
        if skip_scraping:
            logger.error("TMDB data not found and --skip-scraping specified")
            sys.exit(1)

        logger.info("\n" + "="*60)
        logger.info("TMDB data not found - scraping required")
        logger.info("="*60)
        logger.info("\nPlease run TMDB scraping first:")
        logger.info("  python src/scrape_tmdb.py")
        logger.info("\nOr if you have the data, ensure it's at:")
        logger.info(f"  {tmdb_path}")
        sys.exit(1)

    # Load TMDB data
    logger.info("\nLoading TMDB enriched data...")
    df_tmdb = pd.read_csv(tmdb_path)
    logger.info(f"✓ Loaded {len(df_tmdb):,} enriched movies")

    # Merge with TMDB data
    logger.info("\nEnriching with TMDB data...")
    df_enriched = df.merge(
        df_tmdb,
        on='movieId',
        how='left',
        suffixes=('_ml', '_tmdb')
    )

    # Use TMDB title if available, otherwise MovieLens title
    if 'title_tmdb' in df_enriched.columns:
        df_enriched['title'] = df_enriched['title_tmdb'].fillna(df_enriched['title_ml'])
        df_enriched = df_enriched.drop(columns=['title_ml', 'title_tmdb'])

    logger.info(f"✓ Enriched: {len(df_enriched):,} rows")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SOURCING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total ratings: {len(df_enriched):,}")
    logger.info(f"Unique users: {df_enriched['userId'].nunique():,}")
    logger.info(f"Unique movies: {df_enriched['movieId'].nunique():,}")
    logger.info(f"Columns: {len(df_enriched.columns)}")

    # Check enrichment coverage
    if 'director' in df_enriched.columns:
        coverage = (~df_enriched['director'].isna()).sum()
        pct = coverage / len(df_enriched) * 100
        logger.info(f"\nTMDB enrichment coverage: {coverage:,} / {len(df_enriched):,} ({pct:.1f}%)")

    # Save enriched data
    logger.info("\nSaving enriched data...")
    output_path = processed_dir / 'ratings_enriched.csv'
    processed_dir.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(output_path, index=False)
    logger.info(f"✓ Saved to: {output_path}")

    logger.info("\n" + "="*60)
    logger.info("SOURCING COMPLETE")
    logger.info("="*60)
    logger.info("\nNext step: python src/clean_data.py")


def main():
    parser = argparse.ArgumentParser(description='Source and enrich MovieLens data')

    parser.add_argument(
        '--skip-scraping',
        action='store_true',
        help='Skip TMDB scraping (assumes data already exists)'
    )

    args = parser.parse_args()

    try:
        source_data(skip_scraping=args.skip_scraping)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Sourcing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
