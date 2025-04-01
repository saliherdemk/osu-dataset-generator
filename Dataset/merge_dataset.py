import os
import argparse
import pandas as pd


def merge_datasets(file_one, file_two, output_file):
    df_one = pd.read_csv(file_one)
    df_two = pd.read_csv(file_two)

    new_rows = df_two[~df_two["id"].isin(df_one["id"])]
    merged = pd.concat([df_one, new_rows], ignore_index=True)
    merged["unique_id"] = range(len(merged))

    merged.to_csv(output_file, index=False)

    print("Merged file saved on", output_file)


def main():
    parser = argparse.ArgumentParser(description="Merge datasets")
    parser.add_argument(
        "--file_one",
        required=True,
    )
    parser.add_argument(
        "--file_two",
        required=True,
    )
    parser.add_argument(
        "--output_file",
        required=True,
    )

    args = parser.parse_args()

    merge_datasets(args.file_one, args.file_two, args.output_file)


if __name__ == "__main__":
    main()
