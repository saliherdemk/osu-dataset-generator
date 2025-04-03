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

    expected_columns = [
        "type_break",
        "type_circle",
        "type_slider",
        "type_spinner",
        "hit_sound_0",
        "hit_sound_2",
        "hit_sound_4",
        "hit_sound_6",
        "hit_sound_8",
        "hit_sound_10",
        "hit_sound_12",
        "hit_sound_14",
        "sample_set_0.0",
        "sample_set_1.0",
        "sample_set_2.0",
        "sample_set_3.0",
        "effects_0.0",
        "effects_1.0",
        "curve_type_B",
        "curve_type_E",
        "curve_type_L",
        "curve_type_P",
        "new_combo_0",
        "new_combo_1",
    ]
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


def normalize(input_file, output_file, chunk_size=4000000):
    first_chunk = True

    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        print("Normalizing chunk...")

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
