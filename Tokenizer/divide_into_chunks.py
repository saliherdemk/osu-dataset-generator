import argparse
import os
from re import split

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def divide_tokens(df):
    df["end"] = df["time"] + df["duration"]

    grouped = df.groupby("id")

    def assign_chunks_by_interval(group, max_interval=5000):
        chunk_id = 0
        chunk_start = 0
        chunk_ids = []

        for t in group["time"]:
            if t - chunk_start > max_interval:
                chunk_id += 1
                chunk_start = t
            chunk_ids.append(chunk_id)

        group = group.copy()
        group["chunk_id"] = chunk_ids
        return group

    df = df.groupby("id", group_keys=False).apply(assign_chunks_by_interval)

    grouped = df.groupby(["id", "chunk_id"])

    last_chunk_end = 0
    prev_id = None

    chunk_limits = []

    for (id_val, chunk_id), group in grouped:
        if id_val != prev_id:
            last_chunk_end = 0
            prev_id = id_val
        end = group["end"].max()
        chunk_limits.append([id_val, chunk_id, last_chunk_end, end])

        last_chunk_end = end

    cols = ["id", "chunk_id", "delta_time_tick", "duration_tick", "difficulty_rating"]
    limits_df = pd.DataFrame(chunk_limits, columns=["id", "chunk_id", "start", "end"])

    return df[cols], limits_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)

    args = parser.parse_args()

    formatted_df = pd.read_csv(os.path.join(args.input_folder, "formatted.csv"))
    chunked_df, limits_df = divide_tokens(formatted_df)

    chunked_df.to_csv(os.path.join(args.output_folder, "chunked.csv"), index=False)
    limits_df.to_csv(os.path.join(args.output_folder, "chunk_limits.csv"), index=False)


if __name__ == "__main__":
    main()
