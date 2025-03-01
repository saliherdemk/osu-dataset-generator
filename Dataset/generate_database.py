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

    with tqdm(total=len(beatmap_folders), desc="Processing beatmapset folders") as pbar:
        for entry in beatmap_folders:
            entry_path = os.path.join(input_folder, entry)

            osu_files = [
                file for file in os.listdir(entry_path) if file.endswith(".osu")
            ]
            beatmapsetId = entry_path.split("-")[1]

            for index, osu_file in enumerate(osu_files):
                id = beatmapsetId + "-" + str(index)
                processor = BeatmapProcessor(entry_path, osu_file)
                data = processor.get_data()
                data_exporter.write_data(data, id)

            pbar.update(1)

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

            if audio_files:
                first_audio = audio_files[0]
                shutil.copy2(
                    os.path.join(entry_path, first_audio),
                    os.path.join(audio_folder, first_audio),
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
