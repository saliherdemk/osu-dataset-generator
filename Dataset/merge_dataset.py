import os
import argparse
import pandas as pd
import shutil


def merge_datasets(dataset_one, dataset_two):
    csv_files = ["beatmaps.csv", "hit_objects.csv", "timing_points.csv"]

    for file in csv_files:
        file_one = os.path.join(dataset_one, file)
        file_two = os.path.join(dataset_two, file)

        df_one = pd.read_csv(file_one)
        df_two = pd.read_csv(file_two)

        merged_df = pd.concat([df_one, df_two], ignore_index=True)
        merged_df.to_csv(file_one)

    audio_folder_two = os.path.join(dataset_two, "audio")

    for folder in os.listdir(audio_folder_two):
        shutil.copytree(
            os.path.join(audio_folder_two, folder),
            os.path.join(os.path.join(dataset_one, "audio"), folder),
        )

    print(dataset_two, "merged with", dataset_one)


def main():
    parser = argparse.ArgumentParser(description="Merge datasets")
    parser.add_argument(
        "--dataset_one",
        required=True,
    )
    parser.add_argument(
        "--dataset_two",
        required=True,
    )

    args = parser.parse_args()

    merge_datasets(args.dataset_one, args.dataset_two)


if __name__ == "__main__":
    main()
