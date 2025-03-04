import os
import argparse
from tqdm import tqdm
from beatmap_processor import BeatmapProcessor
from data_exporter import DataExporter
import pandas as pd
import shutil


def processed_beatmaps(database_path):
    beatmaps_path = os.path.join(database_path, "beatmaps.csv")
    df = pd.read_csv(beatmaps_path)
    return set(df["ID"].astype(str).str.split("-").str[0])


def process_folder(input_folder, database_path):
    data_exporter = DataExporter(database_path)

    processed = processed_beatmaps(database_path)

    beatmap_folders = [
        entry
        for entry in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, entry))
        and not entry.split("-")[1] in processed
    ]

    skipped_files = []

    with tqdm(total=len(beatmap_folders), desc="Processing beatmapset folders") as pbar:
        for entry in beatmap_folders:
            entry_path = os.path.join(input_folder, entry)

            osu_files = [
                file for file in os.listdir(entry_path) if file.endswith(".osu")
            ]
            beatmapsetId = entry_path.split("-")[-1]

            for index, osu_file in enumerate(osu_files):
                id = beatmapsetId + "-" + str(index)
                processor = BeatmapProcessor(entry_path, osu_file)
                if not processor.is_mode_osu:
                    skipped_files.append(osu_file)
                    continue
                data = processor.get_data()
                data_exporter.write_data(data, id)

            pbar.update(1)
    print(skipped_files)
    print(f"Skipped {len(skipped_files)} beatmaps that modes are not osu.")

    return
    with tqdm(total=len(beatmap_folders), desc="Copying audio files") as pbar:

        for entry in beatmap_folders:
            entry_path = os.path.join(input_folder, entry)
            audio_files = [
                f
                for f in os.listdir(entry_path)
                if f.lower().endswith((".mp3", ".ogg"))
            ]
            audio_folder = os.path.join(database_path, "audio", entry.split("-")[1])
            os.makedirs(audio_folder, exist_ok=True)

            osu_file = [
                f for f in os.listdir(entry_path) if f.lower().endswith((".osu"))
            ][0]

            audio_files = {f.lower(): f for f in os.listdir(entry_path)}

            audio_filename = ""

            with open(os.path.join(entry_path, osu_file), "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("AudioFilename"):
                    audio_filename = line.split(":")[1].strip().lower()
                    continue

            audio_file = audio_files[audio_filename.lower()]

            shutil.copy2(
                os.path.join(entry_path, audio_file),
                os.path.join(audio_folder, audio_file),
            )

            pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Create csv database based on songs folders."
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        help="Path to the folder containing songs folders.",
    )
    parser.add_argument(
        "--database_path",
        required=True,
        help="Database path that will create csv files.",
    )

    args = parser.parse_args()

    process_folder(args.input_folder, args.database_path)


if __name__ == "__main__":
    main()
