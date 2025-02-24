import os
import argparse
from tqdm import tqdm
from beatmap_processor import BeatmapProcessor
from data_exporter import DataExporter


def process_folder(input_folder, database_path):
    data_exporter = DataExporter(database_path)

    beatmap_folders = [
        entry
        for entry in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, entry))
    ]

    with tqdm(total=len(beatmap_folders), desc="Processing beatmap folders") as pbar:
        for entry in beatmap_folders:
            entry_path = os.path.join(input_folder, entry)
            osu_files = [
                file for file in os.listdir(entry_path) if file.endswith(".osu")
            ]

            for index, osu_file in enumerate(
                tqdm(osu_files, desc="Processing", leave=False)
            ):
                id = entry_path.split("-")[1] + "-" + str(index)
                processor = BeatmapProcessor(entry_path, osu_file)
                data_exporter.write(processor.get_data(), id)

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
