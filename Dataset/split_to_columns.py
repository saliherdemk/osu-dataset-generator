import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv


def parse_path(row):
    obj_type = row["type"]
    obj_path = str(row["path"])

    curve_type = "E"
    path = []

    if obj_type == "slider":
        curve_type = obj_path[0]
        coordinates_str = obj_path[2:].split("|")
        path = [list(map(int, point.split(":"))) for point in coordinates_str]

    row["curve_type"] = curve_type
    row["path"] = path
    return row


tqdm.pandas()


def parse_splitted(df):
    max_point_num = df["path"].apply(lambda x: len(x)).max() * 2

    padded_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if len(row["path"]) > 0:
            flattened = np.concatenate(row["path"])
        else:
            flattened = np.array([])

        padded = np.pad(
            flattened,
            (0, max_point_num - len(flattened)),
            mode="constant",
            constant_values=0,
        )

        padded_paths.append(padded)

    column_names = [f"path_{i+1}" for i in range(max_point_num)]

    path_df = pd.DataFrame(padded_paths, columns=column_names, index=df.index)
    return pd.concat([df, path_df], axis=1)


def slience_long_paths(chunk):
    mask = (chunk["path"].apply(lambda x: len(x) > 25)) | (chunk["curve_type"] == "C")
    nan_cols = [
        "x",
        "y",
        "hit_sound",
        "repeat",
        "slider_time",
        "spinner_time",
        "beat_length",
        "meter",
        "slider_velocity",
        "sample_set",
        "volume",
        "effects",
        "difficulty_rating",
    ]
    false_cols = ["new_combo", "has_hit_object"]
    zeros_cols = ["rms"] + [f"mfcc_{i}" for i in range(1, 21)]

    chunk.loc[mask, zeros_cols] = 0
    chunk.loc[mask, "path"] = chunk.loc[mask, "path"].apply(lambda _: [])
    chunk.loc[mask, "type"] = "slient"
    chunk.loc[mask, nan_cols] = np.nan
    chunk.loc[mask, false_cols] = False
    return chunk


def split_to_columns(input_file, output_file, chunk_size=600000):
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    first_chunk = True

    for chunk in reader:

        chunk = chunk.apply(parse_path, axis=1)
        chunk = slience_long_paths(chunk)

        chunk = parse_splitted(chunk)
        chunk.drop(columns=["path"], axis=1, inplace=True)

        mode = "w" if first_chunk else "a"
        header = False
        if first_chunk:
            cols = list(chunk.columns)
            last = int(cols[-1].split("_")[-1])
            for i in range(last + 1, 51):
                new_col = f"path_{i}"
                cols.append(new_col)
                chunk[new_col] = 0
            header = cols
        chunk.to_csv(output_file, mode=mode, header=header, index=False)
        first_chunk = False


def main():
    parser = argparse.ArgumentParser(description="Split curve points")
    parser.add_argument(
        "--input_file",
        required=True,
    )

    parser.add_argument(
        "--output_file",
        required=True,
    )

    args = parser.parse_args()

    split_to_columns(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
