import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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
    df["x"] = df["x"] / 512
    df["y"] = df["y"] / 385
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


def normalize_audio(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["rms"] = scaler.fit_transform(df[["rms"]])

    mfcc_columns = [col for col in df.columns if col.startswith("mfcc")]
    global_max = np.abs(df[mfcc_columns].values).max()
    df[mfcc_columns] = df[mfcc_columns] / global_max

    return df


def normalize_paths(df):
    df = df.fillna(0)

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

    global_max = np.abs(df[path_x_columns].values).max()
    df[path_x_columns] = df[path_x_columns] / global_max

    global_max = np.abs(df[path_y_columns].values).max()
    df[path_y_columns] = df[path_y_columns] / global_max
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


def normalize(input_file, output_file, chunk_size=4000000):
    first_chunk = True

    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        print("Normalizing chunk...")

        chunk = fillnanvalues(chunk)

        chunk = normalize_categoricals(chunk)
        chunk = normalize_nums(chunk)
        chunk = normalize_log(chunk)
        chunk = normalize_audio(chunk)
        chunk = normalize_paths(chunk)

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

    args = parser.parse_args()

    normalize(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
