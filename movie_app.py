import pandas as pd
import ast
import random
import base64
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- TMDB API Key ---
API_KEY = "7a1a62a216447d688f1a306c52e3025b"

# --- Set Background ---
def set_bg(img_path):
    with open(img_path, "rb") as file:
        encoded_img = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Load & Preprocess Data ---
@st.cache_data
def load_files():
    movie_data = pd.read_csv('tmdb_5000_movies.csv')
    credit_data = pd.read_csv('tmdb_5000_credits.csv')
    movie_data = movie_data.merge(credit_data, on='title')
    movie_data = movie_data[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average', 'release_date']]
    movie_data.rename(columns={'id': 'tmdb_id'}, inplace=True)
    movie_data.dropna(inplace=True)
    return movie_data

def extract_names(text):
    return [i['name'] for i in ast.literal_eval(text)][:5]

def get_director_name(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return i['name']
    return ''

def prepare_data(data):
    data['genres'] = data['genres'].apply(extract_names)
    data['keywords'] = data['keywords'].apply(extract_names)
    data['cast'] = data['cast'].apply(extract_names)
    data['crew'] = data['crew'].apply(get_director_name)
    data['overview'] = data['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
    data['crew'] = data['crew'].apply(lambda x: [x])
    data['tags'] = data['overview'] + data['genres'] + data['keywords'] + data['cast'] + data['crew']
    data['tags'] = data['tags'].apply(lambda x: " ".join(x).lower())
    return data

def get_similarity(data):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(data['tags']).toarray()
    return cosine_similarity(vector)

# --- Fetch Poster & Metadata ---
def get_movie_info(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        res = requests.get(url)
        info = res.json()

        poster = f"https://image.tmdb.org/t/p/w500{info.get('poster_path')}" if info.get('poster_path') else None
        budget = info.get('budget', 'Unknown')

        return {
            "poster": poster,
            "budget": f"${budget:,.0f}" if isinstance(budget, int) and budget > 0 else "Unknown"
        }
    except:
        return {"poster": None, "budget": "Unknown"}

# --- Recommendation Functions ---
def suggest_by_title(name, data, sim, min_rate=0, min_year=1900):
    name = name.lower()
    if name not in data['title'].str.lower().values:
        return None
    idx = data[data['title'].str.lower() == name].index[0]
    scores = list(enumerate(sim[idx]))
    sorted_data = sorted(scores, key=lambda x: x[1], reverse=True)[1:]
    result = []
    for i in sorted_data:
        row = data.iloc[i[0]]
        if row['vote_average'] >= min_rate and int(row['release_date'][:4]) >= min_year:
            info = get_movie_info(row['tmdb_id'])
            overview = " ".join(row['overview']) if isinstance(row['overview'], list) else str(row['overview'])
            result.append((
                row['title'],
                row['vote_average'],
                row['release_date'][:4],
                overview,
                info['poster'],
                row['crew'][0],
                row['cast'],
                info['budget']
            ))
        if len(result) == 5:
            break
    return result

def suggest_by_actor(name, data, min_rate=0, min_year=1900):
    name = name.lower()
    matches = data[data['cast'].apply(lambda x: name in [a.lower() for a in x])]
    filtered = matches[
        (matches['vote_average'] >= min_rate) &
        (matches['release_date'].apply(lambda x: int(x[:4]) >= min_year))
    ]
    result = []
    for _, row in filtered.head(5).iterrows():
        info = get_movie_info(row['tmdb_id'])
        overview = " ".join(row['overview']) if isinstance(row['overview'], list) else str(row['overview'])
        result.append((
            row['title'],
            row['vote_average'],
            row['release_date'][:4],
            overview,
            info['poster'],
            row['crew'][0],
            row['cast'],
            info['budget']
        ))
    return result if result else None

def suggest_by_genre(genre, data, min_rate=0, min_year=1900):
    genre = genre.lower()
    matches = data[data['genres'].apply(lambda g: genre in [x.lower() for x in g])]
    filtered = matches[
        (matches['vote_average'] >= min_rate) &
        (matches['release_date'].apply(lambda x: int(x[:4]) >= min_year))
    ]
    result = []
    for _, row in filtered.head(5).iterrows():
        info = get_movie_info(row['tmdb_id'])
        overview = " ".join(row['overview']) if isinstance(row['overview'], list) else str(row['overview'])
        result.append((
            row['title'],
            row['vote_average'],
            row['release_date'][:4],
            overview,
            info['poster'],
            row['crew'][0],
            row['cast'],
            info['budget']
        ))
    return result if result else None

def random_pick(data, col):
    vals = set()
    for row in data[col]:
        vals.update(row if isinstance(row, list) else [])
    return random.sample(list(vals), min(5, len(vals)))

# --- Streamlit App UI ---
st.set_page_config(page_title="Movie Recommendation App", page_icon="🎬")
set_bg("view-black-white-light-projector-theatre.jpg")
st.title("🎬 Movie Recommendation App")

movies = prepare_data(load_files())
sim_scores = get_similarity(movies)

search_type = st.selectbox("Search by", ["Movie Name", "Actor Name", "Genre"])
min_rating = st.slider("Minimum Rating", 0.0, 10.0, 6.0, 0.1)
min_year = st.slider("Minimum Release Year", 1950, 2025, 2010)

if search_type == "Movie Name":
    st.info("Try movies like: " + ", ".join(random.sample(movies['title'].tolist(), 5)))
    user_title = st.text_input("Enter a movie name:")
    if st.button("Recommend"):
        recs = suggest_by_title(user_title, movies, sim_scores, min_rating, min_year)
        if recs:
            st.success("Top Recommendations:")
            for title, rating, year, overview, poster, director, cast, budget in recs:
                if poster:
                    st.image(poster, width=300)
                st.markdown(f"### 🎬 {title}")
                st.markdown(f"**⭐ Rating:** {rating}")
                st.markdown(f"**📅 Year:** {year}")
                st.markdown(f"**🎬 Director:** {director}")
                st.markdown(f"**🎭 Cast:** {', '.join(cast)}")
                st.markdown(f"**💰 Budget:** {budget}")
                st.markdown(f"**📜 Overview:** {overview[:500]}{'...' if len(overview) > 500 else ''}")
                st.markdown("---")
        else:
            st.error("❌ No results found.")

elif search_type == "Actor Name":
    st.info("Try actors like: " + ", ".join(random_pick(movies, 'cast')))
    user_actor = st.text_input("Enter an actor's name:")
    if st.button("Show Movies"):
        recs = suggest_by_actor(user_actor, movies, min_rating, min_year)
        if recs:
            st.success(f"Top movies with {user_actor.title()}:")
            for title, rating, year, overview, poster, director, cast, budget in recs:
                if poster:
                    st.image(poster, width=300)
                st.markdown(f"### 🎬 {title}")
                st.markdown(f"**⭐ Rating:** {rating}")
                st.markdown(f"**📅 Year:** {year}")
                st.markdown(f"**🎬 Director:** {director}")
                st.markdown(f"**🎭 Cast:** {', '.join(cast)}")
                st.markdown(f"**💰 Budget:** {budget}")
                st.markdown(f"**📜 Overview:** {overview[:500]}{'...' if len(overview) > 500 else ''}")
                st.markdown("---")
        else:
            st.error("❌ No results found.")

elif search_type == "Genre":
    genre_pick = st.selectbox("Choose a genre", sorted(list({g for row in movies['genres'] for g in row})))
    if st.button("Find Movies"):
        recs = suggest_by_genre(genre_pick, movies, min_rating, min_year)
        if recs:
            st.success(f"Top {genre_pick.title()} movies:")
            for title, rating, year, overview, poster, director, cast, budget in recs:
                if poster:
                    st.image(poster, width=300)
                st.markdown(f"### 🎬 {title}")
                st.markdown(f"**⭐ Rating:** {rating}")
                st.markdown(f"**📅 Year:** {year}")
                st.markdown(f"**🎬 Director:** {director}")
                st.markdown(f"**🎭 Cast:** {', '.join(cast)}")
                st.markdown(f"**💰 Budget:** {budget}")
                st.markdown(f"**📜 Overview:** {overview[:500]}{'...' if len(overview) > 500 else ''}")
                st.markdown("---")
        else:
            st.error("❌ No results found.")