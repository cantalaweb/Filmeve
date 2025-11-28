# Filmeve Streamlit App - Corporate Demo

**AI-Powered Group Movie Recommendations with Genetic Algorithm Optimization**

## Features

### Core Functionality
- **Login Page**: Flashy gradient design with proof-of-concept authentication
- **20 Personality-Based Users**: Fun Spanish nicknames based on rating behavior
  - "Rosa, la Generosa" (generous rater)
  - "Hugo, el Duro" (harsh critic)
  - And 18 more!
- **Friend Selection**: Visual grid with emoji personalities
- **Compatibility Preview**: See how similar your group's tastes are before running
- **Real-time GA Progress**: Watch fitness evolution across 30 generations
- **Multiple Slates**: Get top 4 optimized movie slates
- **"Try 10 More"**: Cycle through alternative recommendations (max 3 additional slates)

### Enhanced Features

#### 🎬 Recommendations Tab
- **Movie Posters**: Fetched from TMDB API
- **Smart Explanations**: AI explains why each movie works for your group
- **Individual Ratings**: See predicted satisfaction for each friend
- **Filter Options**: Filter by minimum rating and maximum disagreement
- **Star Ratings**: Visual feedback with star displays
- **Bar Charts**: See rating distribution across the group

#### 📊 Group Stats Tab
- **Compatibility Matrix**: Heatmap showing taste similarity
- **Group Insights**:
  - Most Compatible Pair
  - Biggest Critic
  - Most Enthusiastic
  - Hardest to Please
- **Genre Distribution**: Pie chart of recommended genres
- **Algorithm Performance**: Fitness convergence visualization

#### 🎭 Individual Picks Tab
- **Side-by-side Comparison**: What each person wants vs group consensus
- **Trade-off Visualization**: See compromises made for group harmony

#### 📥 Export Tab
- **Download Recommendations**: Export as text file
- **Preview**: See formatted recommendations before downloading

### Technical Features
- **Session State Management**: Persistent data across page changes
- **Lazy Loading**: Model and data loaded once and cached
- **4 Unique Slates**: Genetic algorithm finds diverse solutions
- **Progressive Disclosure**: Expandable sections for details

## Quick Start

```bash
# Navigate to streamlit_app directory
cd streamlit_app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Demo Credentials

**Username**: any
**Password**: any

(Proof of concept - any credentials work!)

## App Flow

1. **Login** → Enter any username/password
2. **Select Friends** → Choose from 20 personalities
   - See compatibility preview for multiple selections
3. **Watch GA Optimize** → Real-time progress bar and fitness chart
4. **Explore Results** → 4 tabs with different views:
   - 🎬 Recommendations (with posters and explanations)
   - 📊 Group Stats (insights and visualizations)
   - 🎭 Individual Picks (personal vs group)
   - 📥 Export (download your slate)
5. **Try More Slates** → Click "Try 10 More Films" up to 3 times
6. **Navigate** → Previous/Next slate, try different friends, or start over

## Corporate Value Propositions

### 1. User Convenience
- Groups find movies everyone enjoys in minutes
- No more endless scrolling or arguments
- AI explains why recommendations work
- Export and share with friends

### 2. Platform Curation
- Same AI can select films for streaming library
- Data-driven content acquisition decisions
- Predict which films will satisfy user base
- Optimize catalog for diverse audience

### 3. Technical Excellence
- Advanced ML with bias reduction
- Genetic algorithm optimization
- Real user preference data
- Scalable architecture

## Key Metrics Displayed

- **Average Group Satisfaction**: Mean predicted rating (1-5 stars)
- **Disagreement Score**: Standard deviation (lower = more consensus)
- **Individual Predictions**: Personalized ratings for each friend
- **Compatibility Matrix**: Taste similarity between friends
- **Fitness Evolution**: Algorithm optimization progress
- **Genre Distribution**: Recommended movie genres

## Technical Stack

- **Frontend**: Streamlit with custom CSS
- **ML Model**: Stacked ensemble (XGBoost, LightGBM, CatBoost, GradientBoosting)
- **Optimization**: Genetic algorithm (30 generations, population 50)
- **Visualization**: Plotly (interactive charts)
- **Data**: MovieLens 100K + TMDB metadata
- **Posters**: TMDB API integration

## File Structure

```
streamlit_app/
├── app.py              # Main application
├── utils.py            # Helper functions
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Performance

- **Model RMSE**: 0.6932 (+4.31% vs baseline)
- **Features**: 106 engineered features
- **GA Runtime**: ~10 minutes for 30 generations
- **Data**: 100,823 bias-reduced ratings
- **Users**: 20 personality-based selections

## Notes

- Requires TMDB API key in `../.env` for poster fetching
- Model and data files must be in `../models/` and `../data/processed/`
- Best viewed on desktop (wide layout)
- Chrome/Firefox recommended

## Future Enhancements

- Real-time collaborative sessions
- Movie trailers embedded
- Social sharing features
- Calendar integration for movie nights
- Mobile-responsive design

---

**Ready to impress Corporate? Run it and watch the magic happen!** ✨

```bash
streamlit run app.py
```
