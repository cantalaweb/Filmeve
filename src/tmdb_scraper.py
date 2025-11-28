"""
TMDB Metadata Scraper for MovieLens 100K Dataset

Fetches additional movie metadata from TMDB API using tmdbId from links.csv.
Enriches movies.csv with: director, cast, country, runtime, budget, revenue, etc.

Usage:
    # Ensure TMDB_API_KEY is in .env file
    # Then run the scraper
    uv run python src/tmdb_scraper.py

    # Or with arguments
    uv run python src/tmdb_scraper.py --limit 100 --output data/processed/movies_enriched.csv

Get a free TMDB API key at: https://www.themoviedb.org/settings/api
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://api.themoviedb.org/3"
USER_AGENT = "Filmeve-ML-Project/1.0 (Educational Data Science Project)"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "MovieLens_100K" / "ml-latest-small"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


class TMDBScraper:
    """Scraper for fetching movie metadata from TMDB API."""

    def __init__(self, api_key: str, delay_range: tuple[float, float] = (0.25, 0.5)):
        """
        Initialize the scraper.

        Args:
            api_key: TMDB API key
            delay_range: Min and max delay between requests (seconds)
        """
        self.api_key = api_key
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            }
        )

    def _make_request(self, endpoint: str, params: dict = None) -> dict | None:
        """Make a request to TMDB API with rate limiting."""
        if params is None:
            params = {}
        params["api_key"] = self.api_key

        url = f"{BASE_URL}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.debug(f"Movie not found: {endpoint}")
                return None
            elif response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get("Retry-After", 10))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self._make_request(endpoint, params)
            else:
                logger.error(f"Error {response.status_code}: {response.text}")
                return None

        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    def get_movie_details(self, tmdb_id: int) -> dict | None:
        """Fetch movie details from TMDB."""
        return self._make_request(f"/movie/{tmdb_id}")

    def get_movie_credits(self, tmdb_id: int) -> dict | None:
        """Fetch movie credits (cast and crew) from TMDB."""
        return self._make_request(f"/movie/{tmdb_id}/credits")

    def fetch_movie_metadata(self, tmdb_id: int) -> dict:
        """
        Fetch complete metadata for a movie.

        Returns dict with: director, cast, country, runtime, budget, revenue,
        release_date, original_language, vote_average, vote_count, overview,
        popularity, tagline
        """
        metadata = {
            "tmdb_id": tmdb_id,
            "director": None,
            "cast": None,
            "country": None,
            "runtime": None,
            "budget": None,
            "revenue": None,
            "release_date": None,
            "original_language": None,
            "vote_average": None,
            "vote_count": None,
            "popularity": None,
            "tagline": None,
            "overview": None,
        }

        # Get movie details
        details = self.get_movie_details(tmdb_id)
        if details:
            metadata["runtime"] = details.get("runtime")
            metadata["budget"] = details.get("budget")
            metadata["revenue"] = details.get("revenue")
            metadata["release_date"] = details.get("release_date")
            metadata["original_language"] = details.get("original_language")
            metadata["vote_average"] = details.get("vote_average")
            metadata["vote_count"] = details.get("vote_count")
            metadata["popularity"] = details.get("popularity")
            metadata["tagline"] = details.get("tagline")
            metadata["overview"] = details.get("overview")

            # Get production countries (first one)
            countries = details.get("production_countries", [])
            if countries:
                metadata["country"] = countries[0].get("iso_3166_1")

        # Add polite delay between requests
        time.sleep(random.uniform(*self.delay_range))

        # Get credits
        credits = self.get_movie_credits(tmdb_id)
        if credits:
            # Get director from crew
            crew = credits.get("crew", [])
            directors = [
                person["name"] for person in crew if person.get("job") == "Director"
            ]
            if directors:
                metadata["director"] = directors[0]  # Primary director

            # Get top 5 cast members
            cast = credits.get("cast", [])
            top_cast = [person["name"] for person in cast[:5]]
            if top_cast:
                metadata["cast"] = "|".join(top_cast)  # Pipe-separated like genres

        # Add polite delay between requests
        time.sleep(random.uniform(*self.delay_range))

        return metadata


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load movies.csv and links.csv from raw data."""
    movies_path = DATA_RAW / "movies.csv"
    links_path = DATA_RAW / "links.csv"

    logger.info(f"Loading movies from {movies_path}")
    movies = pd.read_csv(movies_path)

    logger.info(f"Loading links from {links_path}")
    links = pd.read_csv(links_path)

    return movies, links


def save_checkpoint(data: list[dict], checkpoint_path: Path):
    """Save progress to checkpoint file."""
    with open(checkpoint_path, "w") as f:
        json.dump(data, f)
    logger.info(f"Checkpoint saved: {len(data)} movies")


def load_checkpoint(checkpoint_path: Path) -> list[dict]:
    """Load progress from checkpoint file."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
        logger.info(f"Loaded checkpoint: {len(data)} movies")
        return data
    return []


def main():
    parser = argparse.ArgumentParser(description="Fetch TMDB metadata for MovieLens")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of movies to fetch (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_PROCESSED / "movies_enriched.csv"),
        help="Output file path",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N movies",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    args = parser.parse_args()

    # Get API key from environment (loaded from .env)
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        logger.error(
            "TMDB_API_KEY not found.\n"
            "Add it to your .env file: TMDB_API_KEY=your_key_here\n"
            "Get a free API key at: https://www.themoviedb.org/settings/api"
        )
        return 1

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint file
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    # Load data
    movies, links = load_data()

    # Merge to get tmdb_ids
    merged = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    logger.info(f"Total movies: {len(merged)}")

    # Filter out movies without tmdbId
    valid_movies = merged[merged["tmdbId"].notna()].copy()
    valid_movies["tmdbId"] = valid_movies["tmdbId"].astype(int)
    logger.info(f"Movies with tmdbId: {len(valid_movies)}")

    # Apply limit if specified
    if args.limit:
        valid_movies = valid_movies.head(args.limit)
        logger.info(f"Limited to {args.limit} movies")

    # Load checkpoint if resuming
    metadata_list = []
    processed_ids = set()
    if args.resume:
        metadata_list = load_checkpoint(checkpoint_path)
        processed_ids = {m["tmdb_id"] for m in metadata_list}

    # Initialize scraper
    scraper = TMDBScraper(api_key)

    # Fetch metadata
    total = len(valid_movies)
    start_time = time.time()

    try:
        for idx, row in valid_movies.iterrows():
            tmdb_id = int(row["tmdbId"])
            movie_id = row["movieId"]

            # Skip if already processed
            if tmdb_id in processed_ids:
                continue

            # Fetch metadata
            metadata = scraper.fetch_movie_metadata(tmdb_id)
            metadata["movieId"] = movie_id
            metadata_list.append(metadata)
            processed_ids.add(tmdb_id)

            # Progress logging
            current = len(metadata_list)
            if current % 10 == 0:
                elapsed = time.time() - start_time
                rate = current / elapsed if elapsed > 0 else 0
                eta = (total - current) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {current}/{total} ({current/total*100:.1f}%) "
                    f"- Rate: {rate:.1f}/s - ETA: {eta/60:.1f}min"
                )

            # Save checkpoint
            if current % args.checkpoint_interval == 0:
                save_checkpoint(metadata_list, checkpoint_path)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Saving checkpoint...")
        save_checkpoint(metadata_list, checkpoint_path)

    # Convert to DataFrame
    metadata_df = pd.DataFrame(metadata_list)

    # Merge with original movies data
    enriched = valid_movies.merge(
        metadata_df.drop(columns=["tmdb_id"]), on="movieId", how="left"
    )

    # Save result
    enriched.to_csv(output_path, index=False)
    logger.info(f"Saved enriched data to {output_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file")

    # Summary
    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed/60:.1f} minutes")
    logger.info(f"Total movies enriched: {len(enriched)}")

    # Show sample
    logger.info("\nSample of enriched data:")
    print(enriched[["movieId", "title", "director", "cast", "country"]].head())

    return 0


if __name__ == "__main__":
    exit(main())
