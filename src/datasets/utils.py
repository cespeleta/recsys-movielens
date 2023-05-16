import datetime as dt
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.utils import shuffle


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame]:
    cutoff_dt = df.timestamp.quantile(0.8)
    test = df.query("timestamp > @cutoff_dt")
    train = df.query("timestamp <= @cutoff_dt")
    return train, test


def load_ratings_100k(file: Path | str) -> pd.DataFrame:
    names = ["user_id", "item_id", "rating", "timestamp"]
    data = pd.read_csv(file, names=names, delimiter="\t", header=None)
    data.sort_values(by="timestamp", inplace=True, ascending=True)
    data.reset_index(drop=True, inplace=True)
    data = data.assign(timestamp=lambda x: x.timestamp.apply(dt.datetime.fromtimestamp))
    return data


def load_movie_titles_100k(file: Path | str) -> pd.DataFrame:
    names = ["item_id", "title"]
    movies = pd.read_csv(
        file,
        names=names,
        delimiter="|",
        encoding="latin-1",
        header=None,
        usecols=(0, 1),
    )
    return movies


def add_movie_titles(ratings: pd.DataFrame, titles: pd.DataFrame):
    return ratings.merge(titles, how="left", on="item_id")


def title_item_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the problem with duplicated item_id per title."""
    df["item_id"] = df.groupby("title").item_id.transform("first")
    return df


# def train_test_split(df):
#     ratings = df.copy()
#     ratings = shuffle(ratings)
#     train_size = int(0.75 * ratings.shape[0])
#     train_df = ratings.iloc[:train_size]
#     test_df = ratings.iloc[train_size:]
#     return train_df, test_df


def encode_category(
    df: pd.DataFrame, column: str, save_file: str | Path = None
) -> dict[str, int]:
    values = df[column].unique()
    mapping = {v: int(i) for i, v in enumerate(values)}
    if save_file:
        dump(mapping, save_file)
        print(f"Saving Mapping in file: {save_file}")
    return mapping


def target_scaler(
    df: pd.DataFrame, column: str = "rating", save_file: str | Path = None
) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler = scaler.fit(df[column].values.reshape(-1, 1))
    if save_file:
        dump(scaler, save_file)
        print(f"Saving TargetEncoder in file: {save_file}")
    return scaler
