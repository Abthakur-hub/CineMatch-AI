import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pickle.load(open("models/movies.pkl", "rb"))

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies["content"])

similarity = cosine_similarity(tfidf_matrix)


def recommend(movie_name, top_n=10):

    matches = movies[
        movies["title"].str.lower() == movie_name.lower()
    ]

    if matches.empty:
        return None

    idx = matches.index[0]

    scores = list(enumerate(similarity[idx]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )[1:top_n + 1]

    recommendations = []

    for movie in scores:
        recommendations.append(
            movies.iloc[movie[0]]["title"]
        )

    return recommendations