import argparse
import os
import shutil

import pandas as pd


def merge_datasets(folder_one, folder_two, output_folder):
    df_one = pd.read_csv(os.path.join(folder_one, "encoded.csv"))
    df_two = pd.read_csv(os.path.join(folder_two, "encoded.csv"))

    df_combined = pd.concat([df_one, df_two], ignore_index=True)

    output_audio_folder = os.path.join(output_folder, "audio")
    os.makedirs(output_audio_folder, exist_ok=True)

    df_combined.to_csv(os.path.join(output_folder, "encoded.csv"), index=False)

    audio_path_one = os.path.join(folder_one, "audio")
    audio_path_two = os.path.join(folder_two, "audio")
    audio_folders_one = os.listdir(audio_path_one)
    audio_folders_two = os.listdir(audio_path_two)

    for f in audio_folders_one:
        shutil.copytree(
            os.path.join(audio_path_one, f),
            os.path.join(output_audio_folder, f),
        )

    for f in audio_folders_two:
        shutil.copytree(
            os.path.join(audio_path_two, f),
            os.path.join(output_audio_folder, f),
        )


def main():
    parser = argparse.ArgumentParser(description="Merge datasets")
    parser.add_argument(
        "--folder_one",
        required=True,
    )
    parser.add_argument(
        "--folder_two",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        required=True,
    )

    args = parser.parse_args()

    merge_datasets(args.folder_one, args.folder_two, args.output_folder)


if __name__ == "__main__":
    main()
