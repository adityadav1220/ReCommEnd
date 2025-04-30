import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))

rating_count=ratings["movieId"].value_counts()
min_rating=50
popular_movie=rating_count[rating_count>=min_rating].index
filtered_rating=ratings[ratings["movieId"].isin(popular_movie)]

movies = movies[movies["movieId"].isin(popular_movie)].reset_index(drop=True)

movies["genres_str"] = movies["genres"].apply(lambda x: " ".join(x))

vectorizer= CountVectorizer(tokenizer=lambda x: x.split(), token_pattern=None)
genre_matrix =vectorizer.fit_transform(movies["genres_str"])

cosine_sim=cosine_similarity(genre_matrix)

def get_sim_mov(title, top_n=5):
    if title not in movies["title"].values:
        return f"Error: '{title}' not found in dataset!"
    idx=movies[movies["title"] == title].index[0]
    sim_score=list(enumerate(cosine_sim[idx]))
    sim_score=sorted(sim_score,key=lambda x: x[1], reverse=True)[1:top_n+1]
    similar_movies= [movies.iloc[i[0]]["title"] for i in sim_score]
    return similar_movies

movie_name="Jumanji"
print(get_sim_mov(movie_name))
print(movies["title"].head(100))