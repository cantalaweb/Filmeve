"""
Filmeve - AI-Powered Movie Recommendation for Groups
Enhanced Corporate Demo App
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import time
import joblib
from functools import lru_cache

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import utilities
from utils import (
    get_tmdb_poster, calculate_compatibility_matrix, create_compatibility_heatmap,
    explain_movie_choice, get_genre_distribution, create_genre_pie_chart,
    calculate_group_stats, create_pdf_export, compare_individual_vs_group
)

# Page config
st.set_page_config(
    page_title="Filmeve - Group Movie Recommendations",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .friend-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .movie-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-top: 3px solid #667eea;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stat-highlight {
        background: #f0f4ff;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 3px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Funny user names (20 users with personality)
USER_NAMES = {
    414: ("María", "la Cinéfila"),  # High activity
    599: ("Hugo", "el Duro"),  # Low average rater
    474: ("Carlos", "el Crítico"),  # High activity, decent avg
    105: ("Rosa", "la Generosa"),  # Very high average, low variance
    318: ("Ana", "la Entusiasta"),  # High avg, low variance
    298: ("Pedro", "el Exigente"),  # Very low average
    307: ("Luis", "el Voluble"),  # Low avg, high variance
    489: ("Sara", "la Impredecible"),  # High variance
    68: ("Diego", "el Explorador"),  # High activity
    380: ("Elena", "la Optimista"),  # High average
    606: ("Javier", "el Equilibrado"),  # Good avg, low variance
    249: ("Carmen", "la Constante"),  # High avg, very low variance
    448: ("Miguel", "el Escéptico"),  # Moderate avg, high variance
    182: ("Laura", "la Moderada"),  # Moderate ratings
    603: ("Antonio", "el Apasionado"),  # High variance
    288: ("Isabel", "la Reflexiva"),  # Moderate avg
    610: ("Roberto", "el Positivo"),  # High average
    480: ("Patricia", "la Cambiante"),  # High variance
    387: ("Francisco", "el Analítico"),  # Moderate all
    177: ("Beatriz", "la Versátil"),  # Moderate variance
}

def init_session_state():
    """Initialize session state variables"""
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ''
    if 'selected_friends' not in st.session_state:
        st.session_state.selected_friends = []
    if 'all_slates' not in st.session_state:
        st.session_state.all_slates = None
    if 'current_slate_idx' not in st.session_state:
        st.session_state.current_slate_idx = 0
    if 'fitness_history' not in st.session_state:
        st.session_state.fitness_history = None
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'feature_cols' not in st.session_state:
        st.session_state.feature_cols = None

def show_login_page():
    """Flashy login page"""
    st.markdown('<div class="main-header">Filmeve</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Group Movie Recommendations</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 1, 1.5])

    with col2:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h3>Welcome to the Future of Movie Night</h3>
            <p style='color: #666; font-size: 1.1rem;'>
                Find the perfect movies for you and your friends using advanced AI
                and genetic algorithms. No more endless scrolling or arguments.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Launch Filmeve")

            if submit:
                # Simple auth (proof of concept)
                if username and password:
                    st.session_state.logged_in = True
                    st.session_state.user_name = username
                    st.session_state.page = 'select_friends'
                    st.rerun()
                else:
                    st.error("Please enter both username and password")

        st.markdown("<p style='text-align: center; color: #888; font-size: 0.9rem; margin-top: 1rem;'>Demo credentials: Any username/password will work</p>", unsafe_allow_html=True)

def show_friend_selection():
    """Friend selection page with compatibility preview"""
    st.markdown('<div class="main-header">Select Your Movie Crew</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Welcome, {st.session_state.user_name}</div>', unsafe_allow_html=True)

    st.markdown("### Choose friends to watch movies with:")

    # Display friends in a nice grid
    cols = st.columns(4)
    selected = []

    for idx, (user_id, (first_name, nickname)) in enumerate(USER_NAMES.items()):
        col = cols[idx % 4]
        with col:
            full_name = f"{first_name}, {nickname}"
            if st.checkbox(full_name, key=f"user_{user_id}"):
                selected.append(user_id)

    st.markdown("---")

    if len(selected) > 0:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"### Selected: {len(selected)} friends")

        with col2:
            # Show taste compatibility preview if multiple users
            if len(selected) > 1:
                st.markdown("### Compatibility Preview")
                # Load data for preview
                try:
                    user_data = pd.read_csv('../data/processed/ratings_featured_bias_reduced.csv')
                    compat = calculate_compatibility_matrix(selected, user_data)
                    avg_compat = np.mean([compat[i][j] for i in range(len(selected))
                                         for j in range(i+1, len(selected))])
                    st.metric("Average Compatibility", f"{avg_compat:.0%}")

                    if avg_compat > 0.7:
                        st.success("Very similar tastes")
                    elif avg_compat > 0.5:
                        st.info("Good compatibility")
                    else:
                        st.warning("Diverse tastes - challenge accepted")
                except:
                    pass

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Find Perfect Movies", use_container_width=True):
                st.session_state.selected_friends = selected
                st.session_state.page = 'recommendations'
                st.session_state.current_slate_idx = 0  # Reset slate index
                st.session_state.all_slates = None  # Clear previous results
                st.rerun()
    else:
        st.info("Select at least one friend to continue")

    if st.button("Back to Login"):
        st.session_state.page = 'login'
        st.rerun()

def run_ga_with_progress(user_ids, slate_size=10, n_slates=1):
    """
    Run GA and return best slate
    Returns: (list_of_slates, fitness_history)
    """
    # Load model and data if not already loaded
    if st.session_state.model is None:
        st.session_state.model = joblib.load('../models/model_bias_reduced.pkl')
        st.session_state.feature_cols = joblib.load('../models/feature_columns_bias_reduced.pkl')
        st.session_state.user_data = pd.read_csv('../data/processed/ratings_featured_bias_reduced.csv')

    model = st.session_state.model
    feature_cols = st.session_state.feature_cols
    featured_data = st.session_state.user_data

    # Filter to user data
    user_data = featured_data[featured_data['userId'].isin(user_ids)].copy()

    # Get candidate movies
    movie_counts = user_data.groupby('movieId').size()
    common_movies = movie_counts[movie_counts >= min(2, len(user_ids))].index.tolist()

    if len(common_movies) < slate_size:
        common_movies = user_data['movieId'].unique().tolist()

    # Limit candidate pool
    np.random.seed(42)
    candidate_pool = np.random.choice(common_movies,
                                     size=min(200, len(common_movies)),
                                     replace=False).tolist()

    # GA parameters
    population_size = 50
    n_generations = 30
    mutation_rate = 0.05
    elite_size = 10

    # Progress tracking - centered and compact
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        progress_bar = st.progress(0)
        status_text = st.empty()

    fitness_history = []

    # Initialize population
    population = [np.random.choice(candidate_pool, size=slate_size, replace=False).tolist()
                  for _ in range(population_size)]

    def predict_ratings(slate):
        """Predict ratings for a slate of movies"""
        predictions = {}
        for user_id in user_ids:
            user_preds = []
            for movie_id in slate:
                row = user_data[(user_data['userId'] == user_id) &
                              (user_data['movieId'] == movie_id)]
                if len(row) > 0:
                    # Pass as DataFrame to preserve feature names
                    features = row[feature_cols].iloc[[0]]
                    pred = model.predict(features)[0]
                    user_preds.append(pred)
            predictions[user_id] = user_preds
        return predictions

    def fitness(slate):
        """Calculate fitness of a slate"""
        predictions = predict_ratings(slate)
        user_avgs = [np.mean(preds) if len(preds) > 0 else 2.5
                     for preds in predictions.values()]
        satisfaction = np.mean(user_avgs)
        disagreement = np.std(user_avgs)
        return 0.8 * satisfaction - 0.2 * disagreement

    # Evolution
    for gen in range(n_generations):
        # Evaluate fitness
        fitness_scores = [fitness(slate) for slate in population]
        fitness_history.append(max(fitness_scores))

        # Update progress - simpler, centered display
        progress = (gen + 1) / n_generations
        progress_bar.progress(progress)
        status_text.markdown(
            f"<div style='text-align: center'><strong>Generation {gen + 1}/{n_generations}</strong> | "
            f"Best Fitness: {max(fitness_scores):.3f}</div>",
            unsafe_allow_html=True
        )

        # Elitism - keep top performers
        elite_idx = np.argsort(fitness_scores)[-elite_size:]
        new_population = [population[i].copy() for i in elite_idx]

        # Breed new population
        def tournament_selection(tournament_size=3):
            idx = np.random.choice(population_size, tournament_size, replace=False)
            winner_idx = idx[np.argmax([fitness_scores[i] for i in idx])]
            return population[winner_idx]

        def two_point_crossover(parent1, parent2):
            """Two-point crossover (Order Crossover) matching recommend_for_group.py"""
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

        def mutate(slate):
            """Replacement mutation matching recommend_for_group.py"""
            if np.random.random() < mutation_rate:
                idx = np.random.choice(len(slate))
                available = [m for m in candidate_pool if m not in slate]
                if available:
                    slate[idx] = np.random.choice(available)
            return slate

        # Generate offspring
        while len(new_population) < population_size:
            p1 = tournament_selection()
            p2 = tournament_selection()
            c1, c2 = two_point_crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_population.extend([c1, c2])

        population = new_population[:population_size]
        time.sleep(0.05)  # Visual delay

    # Get top N slates (best unique solutions)
    fitness_scores = [fitness(slate) for slate in population]
    sorted_indices = np.argsort(fitness_scores)[::-1]

    top_slates = []
    for idx in sorted_indices:
        slate = population[idx]
        # Check if this slate is significantly different from existing ones (relaxed threshold)
        is_different = len(top_slates) == 0 or all(len(set(slate) & set(existing['movies'])) <= slate_size * 0.8 for existing in top_slates)

        # Always add if we don't have enough slates yet
        if is_different or len(top_slates) < n_slates:
            # Get movie details
            results = []
            predictions = predict_ratings(slate)

            for movie_id in slate:
                movie_info = user_data[user_data['movieId'] == movie_id].iloc[0]

                # Ensure ALL users have predictions
                user_ratings = {}
                movie_idx = slate.index(movie_id)
                for uid in user_ids:
                    if uid in predictions and movie_idx < len(predictions[uid]):
                        user_ratings[uid] = predictions[uid][movie_idx]
                    else:
                        # Fallback: use global average if no prediction available
                        user_ratings[uid] = 3.5

                rating_values = list(user_ratings.values())
                results.append({
                    'movie_id': movie_id,
                    'title': movie_info['title'],
                    'avg_group_rating': np.mean(rating_values) if rating_values else 3.5,
                    'user_ratings': user_ratings,
                    'disagreement': np.std(rating_values) if len(rating_values) > 1 else 0.0
                })

            # Sort by group rating
            results = sorted(results, key=lambda x: x['avg_group_rating'], reverse=True)

            top_slates.append({
                'movies': slate,
                'recommendations': results,
                'fitness': fitness_scores[idx]
            })

            if len(top_slates) >= n_slates:
                break

    progress_bar.progress(1.0)
    status_text.markdown("<div style='text-align: center'><strong>Optimization Complete</strong></div>", unsafe_allow_html=True)

    return top_slates, fitness_history

def show_recommendations():
    """Show recommendations page with all enhancements"""
    st.markdown('<div class="main-header">Your Perfect Movie Slate</div>', unsafe_allow_html=True)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Recommendations", "Group Stats", "Individual Picks", "Export"])

    with tab1:
        # Show selected friends
        st.markdown("### Your Movie Crew")
        cols = st.columns(min(len(st.session_state.selected_friends), 6))
        for idx, user_id in enumerate(st.session_state.selected_friends):
            col_idx = idx % len(cols)
            first_name, nickname = USER_NAMES[user_id]
            cols[col_idx].markdown(f"<div style='text-align: center'><strong>{first_name}</strong><br><small>{nickname}</small></div>",
                              unsafe_allow_html=True)

        st.markdown("---")

        # Run GA if needed
        if st.session_state.all_slates is None:
            st.markdown("### Finding Optimal Movies with Genetic Algorithm")
            slates, fitness_history = run_ga_with_progress(st.session_state.selected_friends)
            st.session_state.all_slates = slates
            st.session_state.fitness_history = fitness_history

        # Current slate
        current_slate = st.session_state.all_slates[0]
        recommendations = current_slate['recommendations']

        # Display movies
        st.markdown("---")
        st.markdown("### Recommended Movies")

        # Filters
        with st.expander("Filter & Sort Options"):
            col1, col2 = st.columns(2)
            with col1:
                min_rating = st.slider("Minimum Group Rating", 0.0, 5.0, 0.0, 0.5)
            with col2:
                max_disagreement = st.slider("Maximum Disagreement", 0.0, 2.0, 2.0, 0.1)

        # Apply filters
        filtered_movies = [m for m in recommendations
                          if m['avg_group_rating'] >= min_rating and m['disagreement'] <= max_disagreement]

        # Display filtered movies - all expanded
        for idx, movie in enumerate(filtered_movies[:10], 1):
            with st.expander(f"#{idx} - {movie['title']} - {movie['avg_group_rating']:.2f}/5.0", expanded=True):
                col1, col2 = st.columns([1, 4])

                with col1:
                    # Try to get poster - smaller size
                    try:
                        title_clean = movie['title'].split('(')[0].strip()
                        year = None
                        if '(' in movie['title']:
                            year_str = movie['title'].split('(')[1].split(')')[0]
                            if year_str.isdigit():
                                year = int(year_str)

                        poster = get_tmdb_poster(title_clean, year)
                        if poster:
                            st.image(poster, width=240)
                    except:
                        pass

                with col2:
                    st.markdown(f"**Average Group Rating:** {movie['avg_group_rating']:.2f}/5.0")
                    st.markdown(f"**Disagreement:** {movie['disagreement']:.2f}")

                    # Explanation
                    explanation = explain_movie_choice(movie, movie['user_ratings'], USER_NAMES)
                    st.markdown(f"**Why this works:** {explanation}")

                    # Individual ratings - bar chart only, no stars
                    st.markdown("**Individual Predictions:**")

                    # Rating distribution bar chart
                    ratings = list(movie['user_ratings'].values())
                    names = [USER_NAMES[uid][0] for uid in movie['user_ratings'].keys()]

                    fig = go.Figure(data=[go.Bar(
                        x=names,
                        y=ratings,
                        marker_color='#667eea',
                        text=[f"{r:.2f}" for r in ratings],
                        textposition='auto',
                        textfont=dict(size=11),
                        width=0.5  # Bar width
                    )])
                    fig.update_layout(
                        height=220,
                        showlegend=False,
                        yaxis_range=[0, 5.2],
                        yaxis_title="Rating",
                        xaxis_title="",
                        margin=dict(l=30, r=10, t=10, b=30),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=10),
                        bargap=0.3,  # Gap between bars - smaller = closer together
                    )
                    fig.update_xaxes(tickangle=0)
                    fig.update_yaxes(gridcolor='rgba(200,200,200,0.3)')
                    st.plotly_chart(fig, width='stretch', key=f"rating_chart_{movie['movie_id']}")

        # Summary statistics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        # Safe calculation with NaN handling
        ratings_list = [m['avg_group_rating'] for m in recommendations[:10] if not np.isnan(m['avg_group_rating'])]
        disagreement_list = [m['disagreement'] for m in recommendations[:10] if not np.isnan(m['disagreement'])]

        avg_satisfaction = np.mean(ratings_list) if ratings_list else 3.5
        avg_disagreement = np.mean(disagreement_list) if disagreement_list else 0.0

        # Handle empty user_ratings safely
        all_ratings = [r for m in recommendations[:10] for r in m['user_ratings'].values() if not np.isnan(r)]
        min_satisfaction = min(all_ratings) if all_ratings else 0.0
        max_satisfaction = max(all_ratings) if all_ratings else 5.0

        with col1:
            st.markdown(f'<div class="metric-card"><h3>{avg_satisfaction:.2f}/5.0</h3><p>Avg Satisfaction</p></div>',
                       unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{avg_disagreement:.2f}</h3><p>Avg Disagreement</p></div>',
                       unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>{min_satisfaction:.2f}/5.0</h3><p>Minimum Rating</p></div>',
                       unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h3>{max_satisfaction:.2f}/5.0</h3><p>Maximum Rating</p></div>',
                       unsafe_allow_html=True)

        # Buttons
        st.markdown("---")
        button_cols = st.columns([1, 1])

        with button_cols[0]:
            if st.button("Try Different Friends"):
                st.session_state.all_slates = None
                st.session_state.page = 'select_friends'
                st.rerun()

        with button_cols[1]:
            if st.button("Back to Login"):
                st.session_state.all_slates = None
                st.session_state.page = 'login'
                st.rerun()

    with tab2:
        # Group statistics dashboard
        st.markdown("### Group Statistics Dashboard")

        if st.session_state.all_slates:
            current_slate = st.session_state.all_slates[0]
            recommendations = current_slate['recommendations']

            # Calculate stats
            stats = calculate_group_stats(
                st.session_state.selected_friends,
                st.session_state.user_data,
                USER_NAMES,
                recommendations
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Group Insights")
                st.markdown("_Analysis of your group's movie preferences_")

                if 'most_compatible' in stats and stats['most_compatible']:
                    st.info(f"**Most Compatible Pair**\n\n{stats['most_compatible'][0]}\n\nSimilarity: {stats['most_compatible'][1]:.0%}")

                if 'biggest_critic' in stats and stats['biggest_critic']:
                    st.info(f"**Biggest Critic**\n\n{stats['biggest_critic'][0]}\n\nAvg Rating: {stats['biggest_critic'][1]:.2f}/5.0")

                if 'most_enthusiastic' in stats and stats['most_enthusiastic']:
                    st.info(f"**Most Enthusiastic**\n\n{stats['most_enthusiastic'][0]}\n\nAvg Rating: {stats['most_enthusiastic'][1]:.2f}/5.0")

                if 'hardest_to_please' in stats and stats['hardest_to_please']:
                    st.info(f"**Hardest to Please**\n\n{stats['hardest_to_please'][0]}\n\nDeviation: {stats['hardest_to_please'][1]:.2f}")

            with col2:
                # Taste compatibility heatmap
                st.markdown("### Taste Compatibility")
                compat_matrix = calculate_compatibility_matrix(
                    st.session_state.selected_friends,
                    st.session_state.user_data
                )
                fig = create_compatibility_heatmap(
                    st.session_state.selected_friends,
                    USER_NAMES,
                    compat_matrix
                )
                st.plotly_chart(fig, width='stretch', key="compatibility_heatmap")

    with tab3:
        # Individual vs Group comparison
        st.markdown("### Individual Picks vs Group Consensus")

        if st.session_state.all_slates:
            st.markdown("""
            See what each person would pick individually versus what the group algorithm recommends.
            This shows the trade-offs made to optimize for everyone.
            """)

            # Get individual recommendations
            individual_picks = compare_individual_vs_group(
                st.session_state.selected_friends,
                st.session_state.user_data,
                st.session_state.model,
                st.session_state.feature_cols,
                st.session_state.all_slates[0]['recommendations']
            )

            for user_id in st.session_state.selected_friends:
                first_name, nickname = USER_NAMES[user_id]

                with st.expander(f"{first_name}, {nickname}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Top 5 Personal Picks:**")
                        for idx, movie in enumerate(individual_picks[user_id][:5], 1):
                            st.markdown(f"{idx}. {movie['title']} ({movie['predicted_rating']:.2f}/5.0)")

                    with col2:
                        st.markdown("**In Group Recommendations:**")
                        current_slate = st.session_state.all_slates[0]
                        user_group_ratings = [(m['title'], m['user_ratings'].get(user_id, 0))
                                            for m in current_slate['recommendations'][:5]]
                        for idx, (title, rating) in enumerate(user_group_ratings, 1):
                            st.markdown(f"{idx}. {title} ({rating:.2f}/5.0)")

    with tab4:
        # Export recommendations
        st.markdown("### Export Your Recommendations")

        if st.session_state.all_slates:
            current_slate = st.session_state.all_slates[0]

            # Create export text
            export_text = create_pdf_export(
                current_slate['recommendations'],
                st.session_state.selected_friends,
                USER_NAMES
            )

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.download_button(
                    label="Download as Text File",
                    data=export_text,
                    file_name=f"filmeve_recommendations_{st.session_state.user_name}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                st.markdown("---")

                st.markdown("### Preview")
                st.text_area("Recommendations Preview", export_text, height=400)

def main():
    """Main app"""
    init_session_state()

    # Route to appropriate page
    if st.session_state.page == 'login':
        show_login_page()
    elif st.session_state.page == 'select_friends':
        show_friend_selection()
    elif st.session_state.page == 'recommendations':
        show_recommendations()

if __name__ == "__main__":
    main()
