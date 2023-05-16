import datetime as dt
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

# from preparation.io import (
#     add_movie_titles,
#     load_movie_titles_latest,
#     load_ratings_latest,
# )


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame]:
    cutoff_dt = df.timestamp.quantile(0.8)
    test = df.query("timestamp > @cutoff_dt")
    train = df.query("timestamp <= @cutoff_dt")
    return train, test


# TODO: move to io.py
def load_ratings_latest(file: Path | str) -> pd.DataFrame:
    names = ["user_id", "item_id", "rating", "timestamp"]
    data = pd.read_csv(file, names=names, header=0)
    data.sort_values(by="timestamp", inplace=True, ascending=True)
    data = data.assign(timestamp=lambda x: x.timestamp.apply(dt.datetime.fromtimestamp))
    return data


# TODO: move to io.py
def load_movie_titles_latest(file: Path | str) -> pd.DataFrame:
    names = ["item_id", "title", "genres"]
    movies = pd.read_csv(file, names=names, sep=",", header=0)
    movies = movies.assign(genres=lambda x: x.genres.str.split("|"))
    return movies


# TODO: move to io.py
def load_ratings_100k(file: Path | str) -> pd.DataFrame:
    names = ["user_id", "item_id", "rating", "timestamp"]
    data = pd.read_csv(file, names=names, delimiter="\t", header=None)
    data.sort_values(by="timestamp", inplace=True, ascending=True)
    data.reset_index(drop=True, inplace=True)
    data = data.assign(timestamp=lambda x: x.timestamp.apply(dt.datetime.fromtimestamp))
    return data


# TODO: move to io.py
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


# TODO: move to io.py
def add_movie_titles(ratings: pd.DataFrame, titles: pd.DataFrame):
    return ratings.merge(titles, how="left", on="item_id")


def encode_category(
    df: pd.DataFrame, column: str, save_file: str | Path = None
) -> dict[str, int]:
    values = df[column].unique()
    mapping = {v: int(i) for i, v in enumerate(values)}
    if save_file:
        dump(mapping, save_file)
        print(f"Saving Mapping in file: {save_file}")
    return mapping


def main() -> None:
    # path_input = Path("input/ml-latest-small")
    path_input = Path("input/ml-100k")
    print(f"Reading files from folder: {path_input}")
    ratings = load_ratings_100k(path_input / "u.data")
    movie_titles = load_movie_titles_100k(path_input / "u.item")
    ratings = add_movie_titles(ratings, movie_titles)
    train, test = train_test_split(ratings)  # split into train, test
    train, valid = train_test_split(train)  # split into train, valid

    print(
        "Train size: {train} Valid size: {valid} Test size: {test} ".format(
            train=train.shape, valid=valid.shape, test=test.shape
        )
    )
    # Create folder to store datasets and objects
    path_output = Path("output")
    path_datasets = path_output / "datasets"
    path_datasets.mkdir(exist_ok=True)

    # Preprocess categorical columns
    user2int = encode_category(
        train, column="user_id", save_file=path_datasets / "user_encoder.joblib"
    )
    title2int = encode_category(
        train, column="title", save_file=path_datasets / "title_encoder.joblib"
    )
    for df in [train, valid, test]:
        df.loc[:, "user_enc"] = df["user_id"].map(user2int).fillna(-1).astype("int")
        df.loc[:, "title_enc"] = df["title"].map(title2int).fillna(-1).astype("int")

    # Scale target between 0 and 1
    scaler = MinMaxScaler()
    scaler = scaler.fit(train["rating"].values.reshape(-1, 1))
    for df in [train, valid, test]:
        df.loc[:, "rating_scaled"] = scaler.transform(
            df.loc[:, "rating"].values.reshape(-1, 1)
        )

    # Save MinMaxScaler
    file_scaler = path_datasets / "target_scaler.joblib"
    dump(scaler, file_scaler)
    print(f"Saving MinMaxScaler in file: {file_scaler}")

    # Remove new users and movies (items)
    valid = valid.query("user_enc >= 0 and title_enc >= 0")
    test = test.query("user_enc >= 0 and title_enc >= 0")

    # Store DataFrames
    print(f"Saving datasets in folder: {path_datasets}")
    train.to_pickle(path_datasets / "ml_100k_train.pkl")
    valid.to_pickle(path_datasets / "ml_100k_valid.pkl")
    test.to_pickle(path_datasets / "ml_100k_test.pkl")


def main2() -> None:
    # path_input = Path("input/ml-latest-small")
    path_input = Path("input/ml-100k")
    print(f"Reading files from folder: {path_input}")
    train_ratings = load_ratings_100k(path_input / "u1.base")
    test_ratings = load_ratings_100k(path_input / "u1.test")
    movie_titles = load_movie_titles_100k(path_input / "u.item")

    train = add_movie_titles(train_ratings, movie_titles)
    test = add_movie_titles(test_ratings, movie_titles)
    valid = test.copy()

    print(
        "Train size: {train} Valid size: {valid} Test size: {test} ".format(
            train=train.shape, valid=valid.shape, test=test.shape
        )
    )
    # Create folder to store datasets and objects
    path_output = Path("output")
    path_datasets = path_output / "datasets"
    path_datasets.mkdir(exist_ok=True)

    # Preprocess categorical columns
    user2int = encode_category(
        pd.concat([train, valid], axis=1),
        column="user_id",
        save_file=path_datasets / "user_encoder.joblib",
    )
    title2int = encode_category(
        pd.concat([train, valid], axis=1),
        column="title",
        save_file=path_datasets / "title_encoder.joblib",
    )
    for df in [train, valid, test]:
        df.loc[:, "user_enc"] = df["user_id"].map(user2int).fillna(-1)
        df.loc[:, "title_enc"] = df["title"].map(title2int).fillna(-1)

    # # Scale target between 0 and 1
    # scaler = MinMaxScaler()
    # scaler = scaler.fit(train["rating"].values.reshape(-1, 1))
    # for df in [train, valid, test]:
    #     df.loc[:, "rating_scaled"] = scaler.transform(
    #         df.loc[:, "rating"].values.reshape(-1, 1)
    #     )

    # # Save MinMaxScaler
    # file_scaler = path_datasets / "target_scaler.joblib"
    # dump(scaler, file_scaler)
    # print(f"Saving MinMaxScaler in file: {file_scaler}")

    # # Remove new users and movies (items)
    # valid = valid.query("user_enc >= 0 and title_enc >= 0")
    # test = test.query("user_enc >= 0 and title_enc >= 0")

    # Store DataFrames
    print(f"Saving datasets in folder: {path_datasets}")
    train.to_pickle(path_datasets / "ml_100k_train.pkl")
    valid.to_pickle(path_datasets / "ml_100k_valid.pkl")
    test.to_pickle(path_datasets / "ml_100k_test.pkl")


if __name__ == "__main__":
    # TODO: parse args indicanto dataset a leer y ficheros salida
    main()
