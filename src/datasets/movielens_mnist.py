import os
import os.path
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from utils import (
    add_movie_titles,
    load_movie_titles_100k,
    load_ratings_100k,
    title_item_duplicates,
)


class MNIST2(Dataset):
    def __init__(self, root: str | Path) -> None:
        super().__init__()

        if isinstance(root, str):
            root = Path(root).expanduser()

        self.root = root

        if not self._check_exists():
            raise RuntimeError("Dataset not found")

        self.data = self._load_data()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user = self.ratings.user[index]
        movie = self.ratings.item_id[index]
        rating = self.ratings.rating[index]

        return {
            "user": torch.tensor(user, dtype=torch.long),
            "item": torch.tensor(movie, dtype=torch.long),
            "rating": torch.tensor(rating, dtype=torch.float),
        }

    @property
    def target(self):
        return "rating"

    @property
    def file_items(self) -> Path:
        return self.root / "u.item"

    @property
    def file_ratings(self) -> Path:
        return self.root / "u.data"

    def _load_data(self):
        movie_titles = load_movie_titles_100k(self.file_items)
        ratings = load_ratings_100k(self.file_ratings)
        ratings = add_movie_titles(ratings, movie_titles)
        ratings = title_item_duplicates(ratings)
        ratings.reset_index(drop=True, inplace=True)
        return ratings

    def _check_exists(self) -> bool:
        return Path.exists(self.file_items)

    # @property
    # def processed_folder(self) -> str:
    #     # return os.path.join(self.root, self.__class__.__name__, "processed")
    #     return os.path.join("output", "datasets", "ml-100k")

    # @property
    # def str_to_idx(self, col: str) -> dict[str, int]:
    #     values = self.data[col].unique().sort_values()
    #     return {v: i for i, v in enumerate(values)}


# class CIFAR10DataModule(LightningDataModule):
#     transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

#     def train_dataloader(self, *args, **kwargs):
#         trainset = torchvision.datasets.CIFAR10(
#             root=DATASETS_PATH, train=True, download=True, transform=self.transform
#         )
#         return torch.utils.data.DataLoader(
#             trainset, batch_size=2, shuffle=True, num_workers=0
#         )

#     def val_dataloader(self, *args, **kwargs):
#         valset = torchvision.datasets.CIFAR10(
#             root=DATASETS_PATH, train=False, download=True, transform=self.transform
#         )
#         return torch.utils.data.DataLoader(
#             valset, batch_size=2, shuffle=True, num_workers=0
#         )


if __name__ == "__main__":
    # MNIST()
    mm = MNIST2(root="input/ml-100k")
    print(mm)
