import argparse
import pandas as pd
import numpy as np
import librosa


def denormalize_nums(df):
    upper_x = 512
    upper_y = 385
    path_x_columns = [
        col
        for col in df.columns
        if col.startswith("path_") and int(col.split("_")[1]) % 2 != 0
    ]
    path_y_columns = [
        col
        for col in df.columns
        if col.startswith("path_") and int(col.split("_")[1]) % 2 == 0
    ]

    df[path_x_columns] = df[path_x_columns] * upper_x
    df[path_y_columns] = df[path_y_columns] * upper_y
    df["x"] = df["x"] * upper_x
    df["y"] = df["y"] * upper_y
    df["volume"] = df["volume"] * 100
    return df


def denormalize_log(df):
    df["repeat"] = np.expm1(df["repeat"])
    df["slider_time"] = np.expm1(df["slider_time"])
    df["spinner_time"] = np.expm1(df["spinner_time"])
    df["beat_length"] = np.expm1(df["beat_length"])
    df["meter"] = np.expm1(df["meter"])
    df["slider_velocity"] = np.expm1(df["slider_velocity"])
    return df


def denormalize_time(df, audio_path):
    y, sr = librosa.load(audio_path)  # sr=22050
    hop_length = 256
    n_frames = int(np.ceil(len(y) / hop_length))

    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=hop_length
    )
    df["time"] = frame_times * 1000 - df["delta_time"] * -11.61
    return df


def denormalize(input_file, output_file, audio_path):

    df = pd.read_csv(input_file)

    df = denormalize_nums(df)
    df = denormalize_log(df)
    df = denormalize_time(df, audio_path)
    df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Denormalize")

    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--audio_path", required=True)

    args = parser.parse_args()

    denormalize(args.input_file, args.output_file, args.audio_path)


if __name__ == "__main__":
    main()
