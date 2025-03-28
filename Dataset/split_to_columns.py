import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import csv


def parse_path(row):
    x = str(row["path"])
    curve_type = "E"
    spinner_time = 0
    path = []
    if x == "nan":
        spinner_time = 0
    elif not "|" in x:
        spinner_time = x
    else:
        curve_type = x[0]
        coordinates_str = x[2:].split("|")
        path = [list(map(int, point.split(":"))) for point in coordinates_str]

    row["curve_type"] = curve_type
    row["spinner_time"] = spinner_time
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


def parse_mfcc(mfcc_str):
    mfcc_str = mfcc_str.replace("[", "").replace("]", "")
    mfcc_values = re.split(r"\s+", mfcc_str.strip())
    return np.array(mfcc_values, dtype=np.float32)


def split_to_columns(input_file, output_file, chunk_size=10000):
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    first_chunk = True

    for chunk in reader:
        chunk["mfcc"] = chunk["mfcc"].apply(parse_mfcc)

        mfcc_length = 20
        for i in range(mfcc_length):
            chunk[f"mfcc_{i+1}"] = chunk["mfcc"].apply(lambda x: x[i])
        chunk.drop(columns="mfcc", axis=1, inplace=True)

        chunk = chunk.apply(parse_path, axis=1)
        chunk = chunk[chunk["path"].apply(lambda x: len(x) <= 25)]
        chunk = parse_splitted(chunk)
        chunk.drop(columns=["path"], axis=1, inplace=True)

        mode = "w" if first_chunk else "a"
        header = first_chunk
        chunk.to_csv(output_file, mode=mode, header=header, index=False)
        first_chunk = False

    fix(output_file)


def fix(input_file):

    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        rows = list(reader)

    max_columns = max(len(row) for row in rows) - 38
    last_path = int(rows[0][-1].split("_")[1])

    headers = rows[0] + [f"path_{i}" for i in range(last_path + 1, max_columns)]
    rows[0] = headers

    with open(input_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(rows)


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
