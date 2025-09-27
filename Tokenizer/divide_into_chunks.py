import argparse
import os
from re import split

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def divide_tokens(df, chunk_size=10000, step_size=5000):
    results = []

    for doc_id, doc_data in df.groupby("id"):
        max_pos = doc_data["time"].max().astype("int")
        chunk_starts = list(range(0, max_pos + chunk_size, step_size))

        for start in chunk_starts:
            end = start + chunk_size

            chunk_rows = doc_data[
                (doc_data["time"] >= start) & (doc_data["time"] < end)
            ]

            tokenized = ["<chunk_start>"]
            for _, row in chunk_rows.iterrows():
                tokenized.append("<hit_object_start>")
                tokenized.append(f"type_{row['type']}")
                tokenized.append(f"duration_tick_{row['duration_tick']}")
                tokenized.append(f"delta_time_tick{row['delta_time_tick']}")
                tokenized.append("<hit_object_end>")
            tokenized.append("<chunk_end>")
            results.append(
                {
                    "id": doc_id,
                    "chunk_start": start,
                    "chunk_end": end,
                    "tokenized": ",".join(tokenized),
                }
            )
    df = pd.DataFrame(results)
    df["chunk_id"] = df.groupby("id").cumcount()
    df = df[["id", "chunk_id", "chunk_start", "chunk_end", "tokenized"]]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)

    args = parser.parse_args()

    formatted_df = pd.read_csv(os.path.join(args.input_folder, "formatted.csv"))
    chunked_df = divide_tokens(formatted_df)

    chunked_df.to_csv(os.path.join(args.output_folder, "chunked.csv"), index=False)


if __name__ == "__main__":
    main()
