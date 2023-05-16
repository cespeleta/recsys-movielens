from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchvision
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datasets.utils import (
    add_movie_titles,
    encode_category,
    load_movie_titles_100k,
    load_ratings_100k,
    target_scaler,
    title_item_duplicates,
    train_test_split,
)

PATH_RAW_DATA = Path("input")
PATH_PROCESSED_DATA = Path("output/datasets")
PATH_ENCODERS = Path("output/encoders")


class MovieDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = users
        self.items = items
        self.ratings = ratings

    def __len__(self):
        return self.users.shape[0]

    def __getitem__(self, index):
        user = self.users[index]
        movie = self.items[index]
        rating = self.ratings[index]

        return {
            "user": torch.tensor(user, dtype=torch.long),
            "item": torch.tensor(movie, dtype=torch.long),
            "rating": torch.tensor(rating, dtype=torch.float),
        }


class MovielensDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str = "ml-100k",
        target: str = "rating",
        batch_size: int = 64,
    ):
        super().__init__()
        self.dataset = dataset
        self.target = target
        self.batch_size = batch_size
        # To be initialized when running setup()
        self.n_users = None
        self.n_items = None

        self.path_processed_dataset = PATH_PROCESSED_DATA / self.dataset
        self.path_encoders_dataset = PATH_ENCODERS / self.dataset

    def process_raw_data(self) -> None:
        path_raw_dataset = PATH_RAW_DATA / self.dataset
        movie_titles = load_movie_titles_100k(path_raw_dataset / "u.item")
        ratings = load_ratings_100k(path_raw_dataset / "u.data")
        ratings = add_movie_titles(ratings, movie_titles)
        ratings = title_item_duplicates(ratings)

        train, valid = train_test_split(ratings)

        # Preprocess categorical columns
        self.path_encoders_dataset.mkdir(parents=True, exist_ok=True)

        user2int = encode_category(
            train,
            column="user_id",
            save_file=self.path_encoders_dataset / "user_encoder.joblib",
        )
        title2int = encode_category(
            train,
            column="title",
            save_file=self.path_encoders_dataset / "title_encoder.joblib",
        )
        for df in [train, valid]:
            df.loc[:, "user_enc"] = df["user_id"].map(user2int).fillna(-1).astype("int")
            df.loc[:, "title_enc"] = df["title"].map(title2int).fillna(-1).astype("int")

        # Remove new users and movies (items)
        valid = valid.query("user_enc >= 0 and title_enc >= 0")

        # Scale target between 0 and 1
        save_file = self.path_encoders_dataset / "target_encoder.joblib"
        target_encoder = target_scaler(train, column="rating", save_file=save_file)
        for df in [train, valid]:
            df.loc[:, "rating_scaled"] = target_encoder.transform(
                df.loc[:, "rating"].values.reshape(-1, 1)
            )

        # Save processed datasets
        self.path_processed_dataset.mkdir(parents=True, exist_ok=True)
        print(f"Saving datasets in folder: {self.path_processed_dataset}")
        train.to_pickle(self.path_processed_dataset / f"{self.dataset}_train.pkl")
        valid.to_pickle(self.path_processed_dataset / f"{self.dataset}_valid.pkl")
        print("Processing raw data finished")

    def prepare_data(self) -> None:
        # Download dataset, tokenize, save to disk, etc.
        if not self.path_processed_dataset.exists():
            self.process_raw_data()

    def setup(self, stage: Optional[str] = None) -> None:
        # Load pre-processed train / test data
        if not self.path_processed_dataset.exists():
            raise FileNotFoundError(
                f"Dataset not found in {self.path_processed_dataset}"
            )

        df_train = pd.read_pickle(
            self.path_processed_dataset / f"{self.dataset}_train.pkl"
        )
        df_valid = pd.read_pickle(
            self.path_processed_dataset / f"{self.dataset}_valid.pkl"
        )

        # Load processed datasets
        self.n_users = df_train.user_id.nunique()  # 943
        self.n_items = df_train.title_enc.nunique()  # 1625
        assert (
            df_train.item_id.nunique()
            == df_train.title.nunique()
            == df_train.title_enc.nunique()
        ), "Different number of item_id and title."

        # Create Datasets
        self.train_ds = MovieDataset(
            users=df_train.user_enc.values,
            items=df_train.title_enc.values,
            ratings=df_train[self.target].values,
        )
        self.valid_ds = MovieDataset(
            users=df_valid.user_enc.values,
            items=df_valid.title_enc.values,
            ratings=df_valid[self.target].values,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
        )


if __name__ == "__main__":
    dm = MovielensDataModule(dataset="ml-100k", target="rating", batch_size=64)
    dm.setup(stage="fit")
    # print(next(iter(dm.train_dataloader())))
