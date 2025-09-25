import argparse
import os
from re import split

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def divide_tokens(df, max_interval=5000):
    df["end"] = df["time"] + df["duration"]

    grouped = df.groupby("id")

    def assign_chunks_by_interval(group):
        min_time = group["time"].min()
        chunk_id = min_time // max_interval
        chunk_start = chunk_id * max_interval
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

    last_chunk_end = None
    prev_id = None

    chunk_limits = []

    for (id_val, chunk_id), group in grouped:
        if id_val != prev_id:
            last_chunk_end = chunk_id * max_interval
            prev_id = id_val
        end = group["end"].max()
        chunk_limits.append([id_val, chunk_id, last_chunk_end, end])

        last_chunk_end = end

    cols = ["id", "chunk_id", "delta_time_tick", "duration_tick", "difficulty_rating"]
    limits_df = pd.DataFrame(chunk_limits, columns=["id", "chunk_id", "start", "end"])

    grouped = limits_df.groupby("id")
    new_rows = []
    for id_val, group in grouped:
        min_chunk = group["chunk_id"].min()
        while min_chunk != 0:
            new_row = {
                "id": id_val,
                "chunk_id": min_chunk - 1,
                "start": (min_chunk - 1) * max_interval,
                "end": min_chunk * max_interval,
            }
            new_rows.append(new_row)
            min_chunk -= 1

    limits_df = pd.concat([limits_df, pd.DataFrame(new_rows)], ignore_index=True)
    limits_df = limits_df.sort_values(by=["id", "chunk_id"]).reset_index(drop=True)

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
