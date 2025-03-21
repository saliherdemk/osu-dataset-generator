import os
import argparse
import pandas as pd


def merge_datasets(dataset_one, dataset_two, output_file):

    file_one = os.path.join(dataset_one, "hit_objects_formatted.csv")
    file_two = os.path.join(dataset_two, "hit_objects_formatted.csv")

    df_one = pd.read_csv(file_one)
    df_two = pd.read_csv(file_two)

    new_rows = df_two[~df_two["ID"].isin(df_one["ID"])]
    merged = pd.concat([df_one, new_rows], ignore_index=True)
    merged = df_one.sort_values(by="ID").reset_index(drop=True)
    merged["unique_id"] = range(len(df_one))

    merged.to_csv(output_file, index=False)

    print("Merged file saved on", output_file)


def main():
    parser = argparse.ArgumentParser(description="Merge datasets")
    parser.add_argument(
        "--dataset_one",
        required=True,
    )
    parser.add_argument(
        "--dataset_two",
        required=True,
    )
    parser.add_argument(
        "--output_file",
        required=True,
    )

    args = parser.parse_args()

    merge_datasets(args.dataset_one, args.dataset_two, args.output_file)


if __name__ == "__main__":
    main()
