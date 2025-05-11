import argparse
import pandas as pd
import numpy as np
import json
import os


def correct_effect_value(x):
    if x > 8:
        return x != 5746
    return 0 if x == 8 else x


def normalize_categoricals(df, expected_columns):
    df["effects"] = df["effects"].apply(correct_effect_value)
    df["difficulty_rating"] = df["difficulty_rating"].astype(int)

    categorical_columns = [
        "type",
        "hit_sound",
        "sample_set",
        "effects",
        "curve_type",
        "new_combo",
        "difficulty_rating",
        "has_hit_object",
    ]

    df = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns)

    for col in expected_columns:
        if col not in df.columns:
            df[col] = "False"

    df = df[expected_columns]

    return df


def normalize_nums(df):
    upper_x = 512
    upper_y = 385
    path_x_columns = [
        col
        for col in df.columns
        if col.startswith("path_") and int(col.split("_")[1]) % 2 != 0
    ]
    path_y_columns = [
        col
        for col in df.columns
        if col.startswith("path_") and int(col.split("_")[1]) % 2 == 0
    ]
    df[path_x_columns] = df[path_x_columns].clip(lower=0, upper=upper_x)
    df[path_y_columns] = df[path_y_columns].clip(lower=0, upper=upper_y)

    df[path_x_columns] = df[path_x_columns] / upper_x
    df[path_y_columns] = df[path_y_columns] / upper_y

    df["x"] = df["x"] / upper_x
    df["y"] = df["y"] / upper_y
    df["volume"] = df["volume"] / 100
    return df


def normalize_log(df):
    df["repeat"] = np.log1p(df["repeat"])
    df["slider_time"] = np.log1p(df["slider_time"])
    df["spinner_time"] = np.log1p(df["spinner_time"])
    df["beat_length"] = np.log1p(df["beat_length"])
    df["meter"] = np.log1p(df["meter"])
    df["slider_velocity"] = np.log1p(df["slider_velocity"])
    return df


def normalize_audio(df, stats):
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    mfcc_columns = [col for col in df.columns if col.startswith("mfcc")]
    df[mfcc_columns] = (df[mfcc_columns] - mean) / std

    return df


def set_column_dtypes(df):
    strings = ["id", "type", "curve_type"]
    df[strings] = df[strings].astype(str)

    bools = ["new_combo", "has_hit_object"]
    df[bools] = df[bools].astype(bool)

    ints = ["spinner_time"]
    df[ints] = df[ints].astype("int64")

    floats = [col for col in df.columns if col not in strings + bools + ints]
    df[floats] = df[floats].astype("float64")
    return df


def fillnanvalues(df):
    values = {"type": "slient", "new_combo": False}

    df = df.fillna(value=values)
    fill_map = (
        df[df["difficulty_rating"].notna()].groupby("id")["difficulty_rating"].first()
    )
    df["difficulty_rating"] = df["id"].map(fill_map)
    df = df.fillna(0)

    df = set_column_dtypes(df)

    return df


def normalize_times(df):
    margin = -11.61

    df["delta_time"] = (df["frame_time"] - df["time"]) / margin
    df = df.drop(columns=["frame_time", "time", "beatmap_id"])
    return df


def normalize(input_file, output_file, dataset_metadata, chunk_size=50000):
    first_chunk = True

    with open(os.path.join(dataset_metadata, "mfcc.json"), "r") as f:
        stats = json.load(f)

    with open(os.path.join(dataset_metadata, "dataset_columns.json"), "r") as f:
        dataset_cols = json.load(f)

    print("Normalizing chunk...")
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):

        chunk = normalize_times(chunk)
        chunk = fillnanvalues(chunk)
        chunk = normalize_categoricals(chunk, dataset_cols)
        chunk = normalize_nums(chunk)
        chunk = normalize_log(chunk)
        chunk = normalize_audio(chunk, stats)

        chunk.to_csv(output_file, mode="a", index=False, header=first_chunk)
        first_chunk = False

    print("Processing complete.")


def main():
    parser = argparse.ArgumentParser(description="Normalize")
    parser.add_argument(
        "--input_file",
        required=True,
    )

    parser.add_argument(
        "--output_file",
        required=True,
    )

    parser.add_argument("--dataset_metadata", required=True)

    args = parser.parse_args()

    normalize(args.input_file, args.output_file, args.dataset_metadata)


if __name__ == "__main__":
    main()
