import itertools

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.mf import MatrixFactorization

pd.set_option("display.max_columns", None)

# Load model checkpoint
checkpoint = torch.load("output/weights/mf.pt")

model = MatrixFactorization(**checkpoint["kwargs"])
model.load_state_dict(checkpoint["model_state_dict"])
print(model)

# Load data to predict
df_test = pd.read_pickle("output/datasets/ml_latest_small_test.pkl")

user_encoder = checkpoint["user_encoder"]
movie_encoder = checkpoint["movie_encoder"]
df_test["user_enc"] = user_encoder.transform(df_test.user_id.values.reshape(-1, 1))
df_test["movie_enc"] = movie_encoder.transform(df_test.movie_id.values.reshape(-1, 1))
df_test = df_test.query("user_enc >= 0")
df_test = df_test.query("movie_enc >= 0")

# Create dataset
test_ds = MovieDatasetInference(
    users=df_test.user_enc.values,
    movies=df_test.movie_enc.values,
)
batch_size = 64
test_loader = DataLoader(
    dataset=test_ds, batch_size=batch_size, num_workers=0, shuffle=False
)

device = torch.device("cpu")

target_tranformer = checkpoint["target_tranformer"]

# Make predictions
model.eval()
predictions = []
print("Making predictions")
for batch_data in test_loader:
    users = batch_data["user"].to(device)
    movies = batch_data["movie"].to(device)

    y_hat = model(users, movies)
    y_hat = y_hat.detach().numpy().reshape(1, -1)
    preds = target_tranformer.inverse_transform(y_hat)
    predictions.extend(preds.tolist())

predictions = list(itertools.chain(*predictions))
print(predictions)
print("Process finished!")


def process_output(self, output):
    output = output.cpu().detach().numpy()
    return output


# Append predictions to the DataFrame
df_test = df_test.assign(
    predictions=predictions,
    rank=lambda x: x.groupby("user_id").predictions.rank(ascending=False).astype("int"),
)
df_test.head()

# Store predictions


# Ahora hay que anadir las predicciones al dataset y ver cuales han sido los ratings por usuario
# leer los titulos, generos, etc.
