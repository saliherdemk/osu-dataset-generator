import argparse
import pandas as pd
import numpy as np
import json


def get_mfcc_parameters(filepath, output_file):
    df = pd.read_csv(filepath, chunksize=1000000)
    mfcc_cols = [f"mfcc_{i}" for i in range(1, 21)]
    total_sum = np.zeros(len(mfcc_cols))
    total_sq_sum = np.zeros(len(mfcc_cols))
    total_count = 0

    for chunk in df:
        mfcc_data = chunk[mfcc_cols].values
        total_sum += mfcc_data.sum(axis=0)
        total_sq_sum += (mfcc_data**2).sum(axis=0)
        total_count += len(mfcc_data)

    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - (mean**2)
    std = np.sqrt(var + 1e-8)

    stats = {"mean": mean.tolist(), "std": std.tolist()}

    with open(output_file, "w") as f:
        json.dump(stats, f)


def main():
    parser = argparse.ArgumentParser(description="MFCC parameters")
    parser.add_argument(
        "--file",
        required=True,
    )

    parser.add_argument("--output_file", required=True)

    args = parser.parse_args()

    get_mfcc_parameters(args.file, args.output_file)


if __name__ == "__main__":
    main()
