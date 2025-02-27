import os
import argparse
from tqdm import tqdm
from beatmap_processor import BeatmapProcessor
from data_exporter import DataExporter
import pandas as pd


def processed_beatmaps(database_path):
    ids = set()

    with pd.read_csv(
        os.path.join(database_path, "audio.csv"),
        usecols=lambda col: col == "ID",
        on_bad_lines="skip",
        encoding="utf-8",
        chunksize=100,
    ) as reader:
        for chunk in reader:
            ids.update(chunk["ID"])
    return ids


def process_folder(input_folder, database_path):
    data_exporter = DataExporter(database_path)

    processed = processed_beatmaps(database_path)

    beatmap_folders = [
        entry
        for entry in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, entry))
        and int(entry.split("-")[1]) not in processed
    ]

    with tqdm(total=len(beatmap_folders), desc="Processing beatmapset folders") as pbar:
        for entry in beatmap_folders:
            entry_path = os.path.join(input_folder, entry)
            osu_files = [
                file for file in os.listdir(entry_path) if file.endswith(".osu")
            ]
            beatmapsetId = entry_path.split("-")[1]

            audio_data = None

            for index, osu_file in enumerate(osu_files):
                id = beatmapsetId + "-" + str(index)
                processor = BeatmapProcessor(entry_path, osu_file)
                data = processor.get_data()
                audio_data = data["audio"]
                data_exporter.write_data(data, id)

            data_exporter.save_audio(beatmapsetId, audio_data)

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
