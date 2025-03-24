import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import re


def parse_path(row):
    x = row["Path"]
    curve_type = 0
    spinner_time = 0
    path = []
    if pd.isna(x):
        pass
    elif not "|" in x:
        spinner_time = x
    else:
        curve_type = x[0]
        coordinates_str = x[2:].split("|")
        path = [list(map(int, point.split(":"))) for point in coordinates_str]
    row["curve_type"] = curve_type
    row["spinner_time"] = spinner_time
    row["Path"] = path
    return row


tqdm.pandas()


def parse_splitted(df, max_point_num):
    padded_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if len(row["Path"]) > 0:
            flattened = np.concatenate(row["Path"])
        else:
            flattened = np.array([])

        padded = np.pad(
            flattened[:max_point_num],
            (0, max_point_num - min(len(flattened), max_point_num)),
            mode="constant",
        )
        padded_paths.append(padded)

    column_names = [f"Path_{i+1}" for i in range(max_point_num)]

    path_df = pd.DataFrame(padded_paths, columns=column_names, index=df.index)
    return pd.concat([df, path_df], axis=1)


def parse_mfcc(mfcc_str):
    mfcc_str = mfcc_str.replace("[", "").replace("]", "")
    mfcc_values = re.split(r"\s+", mfcc_str.strip())
    return np.array(mfcc_values, dtype=np.float32)


def split_to_columns(input_file, output_file):
    df = pd.read_csv(input_file)

    df = df.apply(parse_path, axis=1)
    max_point_num = df["Path"].apply(lambda x: len(x)).max()
    df = parse_splitted(df, max_point_num)
    df.drop(columns=["Path"], axis=1, inplace=True)

    df["MFCC"] = df["MFCC"].apply(parse_mfcc)
    df.drop(columns="MFCC", axis=1, inplace=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Split MFCC and curve points")
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
