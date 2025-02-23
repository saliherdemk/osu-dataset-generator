import os
import argparse
from beatmap_processor import BeatmapProcessor
from data_exporter import DataExporter


def process_folder(input_folder, database_path):
    data_exporter = DataExporter(database_path)

    for entry in os.listdir(input_folder):
        entry_path = os.path.join(input_folder, entry)

        if os.path.isdir(entry_path):
            print(f"Found folder: {entry}")

            for file in os.listdir(entry_path):
                if file.endswith(".osu"):
                    beatmapset_folder = entry_path
                    osu_file = file
                    processor = BeatmapProcessor(beatmapset_folder, osu_file)
                    data_exporter.write(processor.get_data())


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
