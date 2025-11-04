import argparse
import os
import shutil

import pandas as pd


def merge_datasets(input_folder, output_file):
    files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    first = True

    for f in files:
        path = os.path.join(input_folder, f)
        print(f"Processing {f}")

        df = pd.read_csv(path)

        df.to_csv(output_file, mode="a", index=False, header=first)
        first = False
        del df


def main():
    parser = argparse.ArgumentParser(description="Merge datasets")
    parser.add_argument(
        "--input_folder",
        required=True,
    )

    parser.add_argument(
        "--output_file",
        required=True,
    )

    args = parser.parse_args()

    merge_datasets(args.input_folder, args.output_file)


if __name__ == "__main__":
    main()
