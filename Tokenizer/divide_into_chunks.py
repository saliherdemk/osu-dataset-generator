import argparse

import numpy as np
import pandas as pd


def get_chunks_for_hit(hit_start_ms, duration_ms, chunk_length_ms, chunk_stride_ms):
    hit_end_ms = hit_start_ms + duration_ms
    chunk_starts = np.arange(0, hit_end_ms + chunk_stride_ms, chunk_stride_ms)
    chunk_ids = []
    for i, chunk_start in enumerate(chunk_starts):
        chunk_end = chunk_start + chunk_length_ms
        if hit_end_ms > chunk_start and hit_start_ms < chunk_end:
            chunk_ids.append(i)
    return chunk_ids


def normalize(df_exploded):
    chunk_length_ms = 10000
    v_min, v_max = 0.04, 36.0

    df_exploded["hit_start_rel"] = df_exploded["hit_start_rel"] / chunk_length_ms
    df_exploded["hit_end_rel"] = df_exploded["hit_end_rel"] / chunk_length_ms

    df_exploded["slider_velocity"] = (
        np.log1p(df_exploded["slider_velocity"]) - np.log1p(v_min)
    ) / (np.log1p(v_max) - np.log1p(v_min))

    max_repeat = df_exploded["repeat"].max()
    df_exploded["repeat"] = np.log1p(df_exploded["repeat"]) / np.log1p(max_repeat)

    type_map = {"circle": 0, "slider": 1, "spinner": 2}
    df_exploded["type"] = df_exploded["type"].map(type_map)

    return df_exploded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)

    chunk_length_sec = 10
    overlap_sec = 5

    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    chunk_length_ms = chunk_length_sec * 1000
    chunk_stride_ms = (chunk_length_sec - overlap_sec) * 1000

    df["chunk_id"] = df.apply(
        lambda row: get_chunks_for_hit(
            row["time"], row["duration"], chunk_length_ms, chunk_stride_ms
        ),
        axis=1,
    )

    df_exploded = df.explode("chunk_id").reset_index(drop=True)

    df_exploded["chunk_start"] = df_exploded["chunk_id"] * chunk_stride_ms

    df_exploded["hit_start_rel"] = df_exploded["time"] - df_exploded["chunk_start"]

    df_exploded["hit_end_rel"] = (
        df_exploded["hit_start_rel"] + df_exploded["duration"]
    ).clip(upper=chunk_length_ms)

    df_exploded["hit_start_rel"] = df_exploded["hit_start_rel"].clip(lower=0)

    mask = df_exploded["type"] == "circle"
    df_exploded.loc[mask, "slider_velocity"] = 0.04

    cols = [
        "id",
        "type",
        "repeat",
        "slider_velocity",
        "chunk_id",
        "hit_start_rel",
        "hit_end_rel",
    ]

    df_exploded = df_exploded[cols]

    df_exploded = normalize(df_exploded)

    df_exploded.to_csv(args.output_file, index=False)
    print(f"Updated file saved to {args.input_file}")


if __name__ == "__main__":
    main()
