import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/ml-100k")
OUT_PATH = Path("data/processed")
OUT_PATH.mkdir(parents=True, exist_ok=True)

def preprocess():
    ratings = pd.read_csv(RAW_PATH / "u.data", sep="\t",
        names=["user_id","movie_id","rating","timestamp"])
    users = pd.read_csv(RAW_PATH / "u.user", sep="|",
        names=["user_id","age","gender","occupation","zip_code"])
    movies = pd.read_csv(RAW_PATH / "u.item", sep="|", encoding="latin-1",
        names=["movie_id","title","release_date","video_release","imdb_url"])

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

    ratings.to_csv(OUT_PATH / "ratings.csv", index=False)
    users.to_csv(OUT_PATH / "users.csv", index=False)
    movies.to_csv(OUT_PATH / "movies.csv", index=False)

    matrix = ratings.pivot(index="user_id", columns="movie_id", values="rating")
    matrix.to_csv(OUT_PATH / "ratings_matrix.csv")

if __name__ == "__main__":
    preprocess()
