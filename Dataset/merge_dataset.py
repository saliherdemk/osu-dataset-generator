import argparse
import pandas as pd


def merge_datasets(file_one, file_two, output_file):
    df_one = pd.read_csv(file_one, chunksize=500000)
    df_two = pd.read_csv(file_two, chunksize=500000)

    is_first = True
    for chunk in df_one:
        chunk.to_csv(output_file, mode="a", header=is_first, index=False)
        is_first = False

    for chunk in df_two:
        chunk.to_csv(output_file, mode="a", header=is_first, index=False)


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
