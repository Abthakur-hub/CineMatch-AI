import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("models", exist_ok=True)

movies = pd.read_csv("data/movies.csv")
tags = pd.read_csv("data/tags.csv")

tags = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x.astype(str))).reset_index()

movies = movies.merge(tags, on="movieId", how="left")

movies["tag"] = movies["tag"].fillna("")

movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

movies["content"] = movies["genres"] + " " + movies["tag"]

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies["content"])

similarity = cosine_similarity(tfidf_matrix)

pickle.dump(movies, open("models/movies.pkl", "wb"))
pickle.dump(similarity, open("models/similarity.pkl", "wb"))

print("Model trained successfully!")
print("Movies:", len(movies))
print("Similarity Matrix Shape:", similarity.shape)