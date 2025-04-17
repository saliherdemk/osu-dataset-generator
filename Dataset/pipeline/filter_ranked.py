import os
import argparse
import pandas as pd
import shutil


def filter_ranked_maps(dataset_folder):

    beatmaps_csv = os.path.join(dataset_folder, "beatmaps.csv")
    hit_objects_csv = os.path.join(dataset_folder, "hit_objects.csv")
    time_points_csv = os.path.join(dataset_folder, "timing_points.csv")

    beatmaps_df = pd.read_csv(beatmaps_csv, parse_dates=["ranked_date"])
    hit_objects_df = pd.read_csv(hit_objects_csv)
    time_points_df = pd.read_csv(time_points_csv)

    beatmaps_df = beatmaps_df[
        (beatmaps_df["status"] == "ranked") | (beatmaps_df["status"] == "approved")
    ]

    filtered_beatmaps_df = beatmaps_df[beatmaps_df["ranked_date"] > "2011-01-01"]
    filtered_ids = filtered_beatmaps_df["id"]

    filtered_hit_objects_df = hit_objects_df[hit_objects_df["id"].isin(filtered_ids)]
    filtered_time_points_df = time_points_df[time_points_df["id"].isin(filtered_ids)]

    filtered_beatmaps_df.to_csv(
        os.path.join(dataset_folder, "beatmaps.csv"), index=False
    )
    filtered_hit_objects_df.to_csv(
        os.path.join(dataset_folder, "hit_objects.csv"), index=False
    )
    filtered_time_points_df.to_csv(
        os.path.join(dataset_folder, "timing_points.csv"), index=False
    )

    audio_folder = os.path.join(dataset_folder, "audio")

    removed = []
    for folder in os.listdir(audio_folder):
        if folder not in set([id.split("-")[0] for id in filtered_ids]):
            shutil.rmtree(os.path.join(audio_folder, folder))
            removed.append(folder)
    print(f"Removed {len(removed)} audio file.")


def main():
    parser = argparse.ArgumentParser(description="Filter ranked maps.")
    parser.add_argument(
        "--dataset_folder",
        required=True,
        help="Path to the folder containing dataset folders.",
    )

    args = parser.parse_args()

    filter_ranked_maps(args.dataset_folder)


if __name__ == "__main__":
    main()
