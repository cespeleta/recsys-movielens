import datetime as dt
from pathlib import Path

import pandas as pd


def load_ratings_latest(file: Path | str) -> pd.DataFrame:
    names = ["user_id", "movie_id", "rating", "timestamp"]
    data = pd.read_csv(file, names=names, header=0)
    data.sort_values(by="timestamp", inplace=True, ascending=True)
    data = data.assign(timestamp=lambda x: x.timestamp.apply(dt.datetime.fromtimestamp))
    return data


# 2) Load movie_id and genres
def load_movie_titles_latest(file: Path | str) -> pd.DataFrame:
    names = ["movie_id", "title", "genres"]
    movies = pd.read_csv(file, names=names, sep=",", header=0)
    movies = movies.assign(genres=lambda x: x.genres.str.split("|"))
    return movies


def add_movie_titles(ratings: pd.DataFrame, titles: pd.DataFrame):
    return ratings.merge(titles, how="left", on="movie_id")


# TODO: Convert this functions into a class

# class MovielensLoaderSmall:
#     def load_ratings():
#         pass

#     def load_titles():
#         pass

#     def load_links():
#         pass

#     def load_tags():
#         pass


# # https://stackoverflow.com/questions/55706215/python-design-pattern-using-class-attributes-to-store-data-vs-local-function-v

# class DataManipulator4:
#     def __init__(self, transformer):
#         self._transformer = transformer

#     def run(self, data_sample):
#         data = data_sample.load()
#         results = self._transformer.transform(data)
#         self.data_sample.save(results)


# class DataSample:
#     def __init__(self, filename, connection):
#         self._filename = filename
#         self._connection = connection

#     def load(self):
#         """do stuff to load data, return results"""
#         return read_csv_as_string(self._filename)

#     def save(self, data):
#         """stores string to database"""
#         store_data_to_database(self._connection, data)


# with get_db_connection() as conn:
#     DataManipulator4(Transformer()).run(DataSample("some_file.csv", conn))
