import argparse
import json
import os

import numpy as np
import pandas as pd


def write_mfcc_stats(mean, std, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    with open(output_file, "w") as f:
        json.dump(stats, f)


def write_dataset_columns(difficulty_ratings, max_path, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    columns = (
        [
            "id",
            "x",
            "y",
            "repeat",
            "slider_time",
            "spinner_time",
            "beat_length",
            "meter",
            "slider_velocity",
            "volume",
            "rms",
        ]
        + [f"mfcc_{i}" for i in range(1, 21)]
        + [f"path_{i}" for i in range(1, max_path + 1)]
        + ["delta_time"]
    )
    expected_types = [f"type_{i}" for i in ["circle", "slient", "slider", "spinner"]]
    expected_hitsounds = [f"hit_sound_{i}.0" for i in range(0, 16, 2)]
    expected_sample_set = [f"sample_set_{i}.0" for i in range(4)]
    expected_effects = [f"effects_{i}.0" for i in range(2)]
    expected_curve_types = [f"curve_type_{i}" for i in ["B", "E", "L", "P"]]
    expected_new_combos = [f"new_combo_{i}" for i in ["True", "False"]]
    expected_difficulty_rating = [f"difficulty_rating_{i}" for i in difficulty_ratings]

    expected_columns = (
        columns
        + expected_types
        + expected_hitsounds
        + expected_sample_set
        + expected_effects
        + expected_curve_types
        + expected_new_combos
        + expected_difficulty_rating
    )

    dtypes = ()

    with open(output_file, "w") as f:
        json.dump(expected_columns, f)


def calculate_mfcc_parameters(filepath, output_folder):
    df = pd.read_csv(filepath, chunksize=1000000)
    mfcc_cols = [f"mfcc_{i}" for i in range(1, 21)]

    total_sum = np.zeros(len(mfcc_cols))
    total_sq_sum = np.zeros(len(mfcc_cols))
    total_count = 0
    difficulty_ratings = set()
    max_path = 0

    for chunk in df:
        difficulty_ratings.update(
            chunk[~chunk["difficulty_rating"].isna()]["difficulty_rating"].astype(int)
        )

        path_columns = [col for col in chunk.columns if col.startswith("path_")]
        max_path = max(
            [int(col.split("_")[-1]) for col in path_columns], default=max_path
        )

        mfcc_data = chunk[mfcc_cols].values
        total_sum += mfcc_data.sum(axis=0)
        total_sq_sum += (mfcc_data**2).sum(axis=0)
        total_count += len(mfcc_data)

    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - (mean**2)
    std = np.sqrt(var + 1e-8)

    write_mfcc_stats(mean, std, os.path.join(output_folder, "mfcc.json"))
    write_dataset_columns(
        difficulty_ratings,
        max_path,
        os.path.join(output_folder, "dataset_columns.json"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Get MFCC parameters and dataset columns"
    )
    parser.add_argument(
        "--input_file",
        required=True,
    )

    parser.add_argument("--output_folder", required=True)

    args = parser.parse_args()

    calculate_mfcc_parameters(args.input_file, args.output_folder)


if __name__ == "__main__":
    main()
