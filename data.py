import pandas as pd

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

print("Movies Data:")
print(movies.head(), "\n")

print("Ratings Data:")
print(ratings.head())
