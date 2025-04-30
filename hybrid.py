import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

min_ratings = 50
popular_movies = ratings["movieId"].value_counts()
popular_movies = popular_movies[popular_movies >= min_ratings].index
ratings = ratings[ratings["movieId"].isin(popular_movies)]
movies = movies[movies["movieId"].isin(popular_movies)].reset_index(drop=True)

movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))
movies["genres_str"] = movies["genres"].apply(lambda x: " ".join(x))
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), token_pattern=None)
genre_matrix = vectorizer.fit_transform(movies["genres_str"])
cosine_sim = cosine_similarity(genre_matrix)

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
model = SVD()
model.fit(trainset)

def hybrid_recommendations(user_id, movie_title, top_n=5, alpha=0.7):
    idx = movies[movies["title"] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    hybrid_scores = []
    for i, sim in sim_scores:
        target_movie_id = movies.iloc[i]["movieId"]
        try:
            pred = model.predict(uid=user_id, iid=target_movie_id).est
        except:
            pred = 2.5
        final_score = alpha * sim + (1 - alpha) * (pred / 5.0)
        hybrid_scores.append((i, final_score))

    top_movies = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    filtered_movies = [movies.iloc[i]["title"] for i, _ in top_movies if movies.iloc[i]["title"] != movie_title]
    return filtered_movies[:top_n]

print("🎬 Hybrid Movie Recommender")
try:
    user_id = int(input("Enter user ID (e.g., 1): "))
    movie_title = input("Enter a movie title (e.g., Toy Story (1995)): ")
    alpha = float(input("Enter alpha value (0 = more collaborative, 1 = more content-based, e.g., 0.5): "))

    if movie_title not in movies["title"].values:
        matching_titles = movies[movies["title"].str.contains(movie_title, case=False, na=False)]["title"].tolist()
        
        if matching_titles:
            print(f"\n❓ Did you mean one of these?")
            for i, title in enumerate(matching_titles[:10], 1):
                print(f"{i}. {title}")
            
            try:
                choice = int(input("\nEnter the number of the correct movie (0 to cancel): "))
                if 1 <= choice <= len(matching_titles[:10]):
                    movie_title = matching_titles[choice - 1]
                    recommendations = hybrid_recommendations(user_id, movie_title, top_n=5, alpha=alpha)
                    print(f"\nTop recommendations for user {user_id} based on '{movie_title}':")
                    for i, title in enumerate(recommendations, 1):
                        print(f"{i}. {title}")
                else:
                    print("❌ No movie selected. Exiting.")
            except ValueError:
                print("❌ Invalid input. Exiting.")
        else:
            print(f"\n❌ Movie '{movie_title}' not found in the dataset.")
            print("\n📌 Hint: Here are 10 sample movie titles you can try:")
            print(movies["title"].sample(10).tolist())


except Exception as e:
    print("❌ Error:", e)
