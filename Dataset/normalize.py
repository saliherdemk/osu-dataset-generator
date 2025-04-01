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

    categorical_columns = [
        "type",
        "hit_sound",
        "sample_set",
        "effects",
        "curve_type",
        "new_combo",
    ]
    df = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns)
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


def normalize(input_file, output_file):
    df = pd.read_csv(input_file)
    print("Normalizing...")

    df = normalize_categoricals(df)
    df = normalize_nums(df)
    df = normalize_log(df)
    df = normalize_audio(df)
    df = normalize_paths(df)

    df.to_csv(output_file, index=False)


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
