import os
import argparse
from tqdm import tqdm
import pandas as pd
import shutil
import subprocess
from tqdm import tqdm


def remove_rows_by_ids(dataset_folder, ids_to_remove):
    beatmaps_csv = os.path.join(dataset_folder, "beatmaps.csv")
    hit_objects_csv = os.path.join(dataset_folder, "hit_objects.csv")
    time_points_csv = os.path.join(dataset_folder, "timing_points.csv")

    beatmaps_df = pd.read_csv(beatmaps_csv)
    hit_objects_df = pd.read_csv(hit_objects_csv)
    time_points_df = pd.read_csv(time_points_csv)

    beatmaps_df["beatmap_id"] = beatmaps_df["ID"].str.split("-").str[0]
    hit_objects_df["beatmap_id"] = hit_objects_df["ID"].str.split("-").str[0]
    time_points_df["beatmap_id"] = time_points_df["ID"].str.split("-").str[0]

    beatmaps_df = beatmaps_df[~beatmaps_df["beatmap_id"].isin(ids_to_remove)]
    hit_objects_df = hit_objects_df[~hit_objects_df["beatmap_id"].isin(ids_to_remove)]
    time_points_df = time_points_df[~time_points_df["beatmap_id"].isin(ids_to_remove)]

    beatmaps_df.to_csv(beatmaps_csv, index=False)
    hit_objects_df.to_csv(hit_objects_csv, index=False)
    time_points_df.to_csv(time_points_csv, index=False)

    print(f"Removed rows with IDs {ids_to_remove} from the CSV files.")


def clear_corrupted_files(folder_path):
    files = os.listdir(folder_path)
    if len(files) > 1:
        for file in files:
            if not file.startswith("fixed."):
                file_path = os.path.join(folder_path, file)
                os.remove(file_path)


def fix_bom_issue(folder_path):
    file_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            file_path,
            "-c",
            "copy",
            os.path.join(folder_path, "fixed.mp3"),
        ],
        check=True,
    )


def fix_header_issue(folder_path):
    file_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            file_path,
            "-acodec",
            "libmp3lame",
            "-b:a",
            "192k",
            os.path.join(folder_path, "fixed.mp3"),
        ],
        check=True,
    )


def is_audio_corrupted(folder_path):
    try:
        filename = os.listdir(folder_path)[0]
        file_path = os.path.join(folder_path, filename)
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                os.path.join(folder_path, file_path),
                "-f",
                "null",
                "-",
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )
        if result.stderr:
            print(f"Corruption detected in {file_path}:\n{result.stderr.decode()}")
            return True
        return False
    except FileNotFoundError:
        return True


def fix_corrupted_audios(dataset_folder):
    audio_path = os.path.join(dataset_folder, "audio")
    corrupted = []

    for folder in tqdm(os.listdir(audio_path), desc="Checking audio files"):
        folder_path = os.path.join(audio_path, folder)
        if is_audio_corrupted(folder_path):
            corrupted.append(folder_path)

    for beatmap_path in tqdm(corrupted, desc="Fixing Header issue"):
        try:
            fix_header_issue(beatmap_path)
            clear_corrupted_files(beatmap_path)
        except:
            continue

    remaining = []
    for beatmap_path in tqdm(corrupted, desc="Checking remaining files."):
        if is_audio_corrupted(beatmap_path):
            remaining.append(beatmap_path)

    for beatmap_path in tqdm(remaining, desc="Fixing BOM issue"):
        try:
            fix_bom_issue(beatmap_path)
            clear_corrupted_files(beatmap_path)
        except:
            continue

    cant_fix = []
    for beatmap_path in tqdm(remaining, desc="Checking remaining files."):
        if is_audio_corrupted(beatmap_path):
            cant_fix.append(beatmap_path)
    cant_fix_ids = [path.split("/")[-1] for path in cant_fix]

    for beatmap_id in cant_fix_ids:
        shutil.rmtree(os.path.join(audio_path, beatmap_id))

    print(f"{len(cant_fix_ids)} beatmaps couldn't fix. Removing...")
    remove_rows_by_ids(dataset_folder, cant_fix_ids)


def main():
    parser = argparse.ArgumentParser(description="Fix corrupted audios.")
    parser.add_argument(
        "--dataset_folder",
        required=True,
        help="Path to the folder containing dataset folders.",
    )

    args = parser.parse_args()

    fix_corrupted_audios(args.dataset_folder)


if __name__ == "__main__":
    main()
