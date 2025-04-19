import argparse
import pandas as pd
import numpy as np
import json


def correct_effect_value(x):
    if x > 8:
        return x != 5746
    return 0 if x == 8 else x


def normalize_categoricals(df):
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

    expected_types = [f"type_{i}" for i in ["circle", "slient", "slider", "spinner"]]
    expected_hitsounds = [f"hit_sound_{i}" for i in range(0, 16, 2)]
    expected_sample_set = [f"sample_set_{i}.0" for i in range(4)]
    expected_effects = [f"effects_{i}.0" for i in range(2)]
    expected_curve_types = [f"curve_type_{i}" for i in ["B", "E", "L", "P"]]
    expected_new_combos = [f"new_combo_{i}" for i in range(2)]
    expected_difficulty_rating = [f"difficulty_rating_{i}" for i in range(13)]

    expected_columns = (
        expected_types
        + expected_hitsounds
        + expected_sample_set
        + expected_effects
        + expected_curve_types
        + expected_new_combos
        + expected_difficulty_rating
    )
    df = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns)

    for col in expected_columns:
        if col not in df.columns:
            df[col] = "False"
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

    df["x"] = df["x"] / upper_x
    df["y"] = df["y"] / upper_y
    df["volume"] = df["volume"] / 100
    df["slider_velocity"] = df["slider_velocity"] * -1
    return df


def normalize_log(df):
    df["repeat"] = np.log1p(df["repeat"])
    df["length"] = np.log1p(df["length"])
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

    ints = ["beatmap_id", "spinner_time"]
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


def normalize(input_file, output_file, mfcc_parameters, chunk_size=4000000):
    first_chunk = True
    with open(mfcc_parameters, "r") as f:
        stats = json.load(f)

    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        print("Normalizing chunk...")

        chunk = fillnanvalues(chunk)

        chunk = normalize_categoricals(chunk)
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

    parser.add_argument("--mfcc_parameters", required=True)

    args = parser.parse_args()

    normalize(args.input_file, args.output_file, args.mfcc_parameters)


if __name__ == "__main__":
    main()
