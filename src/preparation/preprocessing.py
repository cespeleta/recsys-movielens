# Encode categorical columns

from pathlib import Path

import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

set_config(transform_output="pandas")
# Montar pipeline de sklearn y que me devuelva el target y la variables
# transformadas


def encode_categories(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    encoder = OrdinalEncoder(
        dtype="int", handle_unknown="use_encoded_value", unknown_value=-1
    )
    ct = ColumnTransformer([("ordinal_encoder", encoder, cat_columns)])
    df_transformed = ct.fit(df)
    df_transformed.columns = [f"{c}_encoded" for c in columns]
    return df_transformed

    Path()
    print("Encoder stored in: {output/}")


if __name__ == "__main__":
    cat_columns = ["user_id", "movie_id"]
    target_column = ["rating"]

    encoder = OrdinalEncoder(
        dtype="int", handle_unknown="use_encoded_value", unknown_value=-1
    )
    ct = ColumnTransformer(
        [
            ("ordinal_encoder", encoder, cat_columns),
            # ("ratings_scaler", MinMaxScaler(), target_column),
        ]
    )

    df_train = pd.read_pickle("output/datasets/ml_latest_small_train.pkl")
    print(ct.fit_transform(df_train).columns)
