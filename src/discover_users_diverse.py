#!/usr/bin/env python3
"""
Diverse User Discovery: Embrace varied tastes!

This approach accepts that users have diverse preferences.
No genre-forcing, no artificial constraints.

Focus on:
- User rating patterns (generous vs critical)
- Director/actor preferences
- Activity levels (prolific vs selective)
- Natural viewing patterns
"""
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
from pathlib import Path

print("="*80)
print("DIVERSE USER DISCOVERY: Embracing Varied Tastes")
print("="*80)

# Load data
print("\n[1/4] Loading data...")
# Use ratings_cleaned.csv - has all the columns we need for user discovery
df_ratings = pd.read_csv('data/processed/ratings_cleaned.csv')[['userId', 'movieId', 'rating', 'title', 'genres']]
df_movies = pd.read_csv('data/processed/movies_enriched.csv')

print(f"✓ Loaded {len(df_ratings):,} ratings from {df_ratings['userId'].nunique():,} users")
print(f"✓ Loaded {len(df_movies):,} movies")


def analyze_user_rating_pattern(user_id):
    """Analyze how a user rates movies."""
    user_ratings = df_ratings[df_ratings['userId'] == user_id]

    return {
        'userId': int(user_id),
        'n_ratings': len(user_ratings),
        'mean_rating': float(user_ratings['rating'].mean()),
        'std_rating': float(user_ratings['rating'].std()),
        'n_high_ratings': int((user_ratings['rating'] >= 4.5).sum()),
        'pct_high_ratings': float((user_ratings['rating'] >= 4.5).mean()),
        'n_low_ratings': int((user_ratings['rating'] <= 2.0).sum()),
        'rating_range': float(user_ratings['rating'].max() - user_ratings['rating'].min()),
    }


def find_users_by_keyword(keyword_list, category_name,
                          min_movies=3, min_rating=4.0):
    """
    Find users who rated movies matching keywords highly.
    NO exclusivity requirements - embrace diverse tastes!
    """
    print(f"\n{'='*70}")
    print(f"Finding: {category_name}")
    print(f"{'='*70}")
    print(f"Keywords: {', '.join(keyword_list[:3])}...")
    print(f"Criteria: ≥{min_movies} movies, avg rating ≥{min_rating}")

    pattern = '|'.join(keyword_list)

    # Get movies matching keywords
    matching_movies = df_movies[
        df_movies['title'].str.contains(pattern, case=False, na=False)
    ]

    print(f"  Found {len(matching_movies)} matching movies")

    # Get ratings for these movies
    matching_ratings = df_ratings[df_ratings['movieId'].isin(matching_movies['movieId'])]

    # Aggregate by user
    user_stats = matching_ratings.groupby('userId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    user_stats.columns = ['userId', 'avg_rating', 'count']

    # Filter by criteria (relaxed - no exclusivity!)
    candidates = user_stats[
        (user_stats['count'] >= min_movies) &
        (user_stats['avg_rating'] >= min_rating)
    ]

    print(f"  ✓ Found {len(candidates)} users")

    if len(candidates) == 0:
        print(f"  ⚠️  No users found - skipping")
        return None

    # Sort by rating count (most engaged first)
    candidates = candidates.sort_values('count', ascending=False)

    # Take top 10
    selected = candidates.head(10)

    print(f"  → Selected top {len(selected)} users:")
    for _, user in selected.head(5).iterrows():
        print(f"     User {int(user['userId'])}: {int(user['count'])} movies, "
              f"avg={user['avg_rating']:.2f}")
    if len(selected) > 5:
        print(f"     ... and {len(selected) - 5} more")

    return {
        'category_name': category_name,
        'description': f"Users who enjoy {category_name.lower()}",
        'users': [int(uid) for uid in selected['userId'].tolist()],
        'user_details': [
            {
                'userId': int(row['userId']),
                'count': int(row['count']),
                'avg_rating': float(row['avg_rating'])
            }
            for _, row in selected.iterrows()
        ]
    }


def find_users_by_rating_pattern(pattern_type):
    """
    Find users based on rating patterns.

    Patterns:
    - generous: Rate most movies highly (mean ≥ 4.0)
    - critical: Selective, use full rating range
    - prolific: Rate many movies (≥100)
    """
    print(f"\n{'='*70}")
    print(f"Finding: {pattern_type} users")
    print(f"{'='*70}")

    all_users = df_ratings['userId'].unique()
    user_patterns = [analyze_user_rating_pattern(uid) for uid in all_users]

    if pattern_type == 'generous':
        # Users who rate most movies highly
        criteria = [
            u for u in user_patterns
            if u['mean_rating'] >= 4.0 and u['n_ratings'] >= 50
        ]
        criteria.sort(key=lambda x: x['mean_rating'], reverse=True)
        description = "Generous raters who enjoy most movies"

    elif pattern_type == 'critical':
        # Users who are selective (lower mean, higher std)
        criteria = [
            u for u in user_patterns
            if u['mean_rating'] < 3.5 and u['std_rating'] > 1.0 and u['n_ratings'] >= 50
        ]
        criteria.sort(key=lambda x: x['std_rating'], reverse=True)
        description = "Critical raters who are selective"

    elif pattern_type == 'prolific':
        # Users who rate many movies
        criteria = [
            u for u in user_patterns
            if u['n_ratings'] >= 100
        ]
        criteria.sort(key=lambda x: x['n_ratings'], reverse=True)
        description = "Prolific raters with many ratings"

    else:
        return None

    print(f"  ✓ Found {len(criteria)} {pattern_type} users")

    if len(criteria) == 0:
        print(f"  ⚠️  No users found - skipping")
        return None

    selected = criteria[:10]

    print(f"  → Selected top {len(selected)} users:")
    for user in selected[:5]:
        print(f"     User {user['userId']}: {user['n_ratings']} ratings, "
              f"mean={user['mean_rating']:.2f}, std={user['std_rating']:.2f}")
    if len(selected) > 5:
        print(f"     ... and {len(selected) - 5} more")

    return {
        'category_name': f"{pattern_type.title()} Raters",
        'description': description,
        'users': [u['userId'] for u in selected],
        'user_details': selected
    }


# ============================================================================
# MAIN DISCOVERY
# ============================================================================

print("\n[2/4] Discovering user categories...")
print("="*80)

discovered_categories = []

# Rating pattern-based
for pattern in ['generous', 'critical', 'prolific']:
    result = find_users_by_rating_pattern(pattern)
    if result:
        discovered_categories.append(result)

print("\n[3/4] Discovering director/actor fans...")
print("="*80)

# Director fans (RELAXED criteria - no exclusivity requirement!)
directors = {
    'Steven Spielberg': ['Jurassic', 'Schindler', 'Saving Private Ryan', 'E.T.',
                        'Jaws', 'Indiana Jones', 'Close Encounters', 'War Horse',
                        'Lincoln', 'Minority Report', 'Catch Me If You Can'],
    'Christopher Nolan': ['Inception', 'Dark Knight', 'Interstellar', 'Prestige',
                         'Memento', 'Batman Begins', 'Dunkirk'],
    'Quentin Tarantino': ['Pulp Fiction', 'Kill Bill', 'Django', 'Inglourious',
                         'Reservoir Dogs', 'Jackie Brown', 'Hateful Eight'],
    'Ridley Scott': ['Alien', 'Blade Runner', 'Gladiator', 'Martian',
                    'Thelma', 'Black Hawk Down'],
    'Martin Scorsese': ['Goodfellas', 'Taxi Driver', 'Departed', 'Casino',
                       'Raging Bull', 'Shutter Island', 'Wolf of Wall Street'],
}

for director, movies in directors.items():
    result = find_users_by_keyword(
        movies,
        f"{director} Fans",
        min_movies=3,
        min_rating=4.0  # Lowered from 4.5
    )
    if result:
        discovered_categories.append(result)

# Classic directors
classic_directors = {
    'Alfred Hitchcock': ['Psycho', 'Vertigo', 'Rear Window', 'North by Northwest',
                        'Birds', 'Rope'],
    'Stanley Kubrick': ['2001', 'Clockwork Orange', 'Shining', 'Full Metal Jacket',
                       'Dr. Strangelove', 'Eyes Wide Shut'],
}

for director, movies in classic_directors.items():
    result = find_users_by_keyword(
        movies,
        f"{director} Fans",
        min_movies=2,  # Even more relaxed
        min_rating=4.0
    )
    if result:
        discovered_categories.append(result)

# Popular franchises
franchises = {
    'Star Wars Fans': ['Star Wars', 'Empire Strikes', 'Return of the Jedi',
                      'Phantom Menace', 'Attack of the Clones', 'Force Awakens'],
    'Marvel Fans': ['Iron Man', 'Avengers', 'Captain America', 'Thor',
                   'Guardians', 'Spider-Man', 'Black Panther'],
    'Lord of the Rings Fans': ['Fellowship', 'Two Towers', 'Return of the King',
                              'Hobbit'],
}

for franchise, movies in franchises.items():
    result = find_users_by_keyword(
        movies,
        franchise,
        min_movies=3,
        min_rating=4.0
    )
    if result:
        discovered_categories.append(result)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n[4/4] Saving results...")
print("="*80)

output_path = Path('data/processed/discovered_users_diverse.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(discovered_categories, f, indent=2)

print(f"\n✓ Saved {len(discovered_categories)} categories to {output_path}")
print(f"\nCategories discovered:")
for cat in discovered_categories:
    print(f"  • {cat['category_name']}: {len(cat['users'])} users")

total_users = len(set(u for cat in discovered_categories for u in cat['users']))
print(f"\n✓ Total unique users: {total_users}")

# Create test groups
print("\n" + "="*80)
print("Creating test groups...")
print("="*80)

test_groups = []

# Create groups from categories
for cat in discovered_categories:
    if len(cat['users']) >= 3:
        # Take 5 users per group
        test_groups.append({
            'name': cat['category_name'],
            'description': cat['description'],
            'users': cat['users'][:5]
        })
        print(f"  ✓ {cat['category_name']}: {len(cat['users'][:5])} users")

# Save test groups
groups_path = Path('data/processed/test_groups_diverse.json')
with open(groups_path, 'w') as f:
    json.dump(test_groups, f, indent=2)

print(f"\n✓ Created {len(test_groups)} test groups")
print(f"✓ Saved to {groups_path}")

print("\n" + "="*80)
print("DIVERSE DISCOVERY COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  • Categories: {len(discovered_categories)}")
print(f"  • Test groups: {len(test_groups)}")
print(f"  • Total users: {total_users}")
print(f"\nPhilosophy:")
print(f"  ✓ Embrace diverse tastes - users can like multiple genres!")
print(f"  ✓ No artificial genre constraints")
print(f"  ✓ Focus on genuine user patterns (rating behavior, favorites)")
print(f"  ✓ Evaluate on fairness, diversity, and satisfaction")
print(f"\nNext: Test with ORIGINAL recommender (no genre tricks)!")
