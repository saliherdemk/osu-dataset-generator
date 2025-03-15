import os
import argparse
import pandas as pd
import shutil


def format_dataset(dataset_path):
    beatmaps_csv = os.path.join(dataset_path, "beatmaps.csv")
    hit_objects_csv = os.path.join(dataset_path, "hit_objects.csv")
    time_points_csv = os.path.join(dataset_path, "timing_points.csv")

    beatmaps_df = pd.read_csv(beatmaps_csv, parse_dates=["ranked_date"])
    hit_objects_df = pd.read_csv(hit_objects_csv)
    time_points_df = pd.read_csv(time_points_csv)


def main():
    parser = argparse.ArgumentParser(description="Merge datasets")
    parser.add_argument(
        "--dataset_path",
        required=True,
    )
    args = parser.parse_args()

    format_dataset(args.dataset_path)


if __name__ == "__main__":
    main()
