import os
import argparse
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import librosa
import ast


class Formatter:
    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path

    def process(self, series):
        audio_path = os.path.join(self.dataset_path, "audio")

        audio_folder = os.path.join(audio_path, series["beatmap_id"].iloc[0])
        audio_file = os.path.join(audio_folder, os.listdir(audio_folder)[0])

        y, sr = librosa.load(audio_file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        rms_energy = librosa.feature.rms(y=y)
        audio_features = []
        for _, row in series.iterrows():
            id = row["ID"]
            break_points = ast.literal_eval(row["BreakPoints"])
            for point in break_points:
                _, start, end = point.split(",")
                start_frame = librosa.time_to_frames(float(start) / 1000, sr=sr)
                end_frame = librosa.time_to_frames(float(end) / 1000, sr=sr)

                for frame_idx in range(start_frame, end_frame):
                    mfcc_segment = mfcc[:, frame_idx]
                    rms_segment = rms_energy[:, frame_idx]
                    frame_time = librosa.frames_to_time(frame_idx, sr=sr)

                    audio_features.append(
                        [
                            id,
                            frame_time,
                            "break",
                            mfcc_segment,
                            rms_segment,
                            id.split("-")[0],
                        ]
                    )
        return pd.DataFrame(
            audio_features, columns=["ID", "Time", "Type", "MFCC", "RMS", "beatmap_id"]
        )

    def safe_process(self, group):
        try:
            return self.process(group)
        except Exception as e:
            print(f"Error processing group: {set(group["ID"])}, Error: {e}")
            return None

    def format_dataset(self):
        beatmaps_df = pd.read_csv(os.path.join(self.dataset_path, "beatmaps.csv"))

        has_break_points = beatmaps_df[beatmaps_df["BreakPoints"] != "[]"]
        has_break_points = has_break_points[["ID", "BreakPoints"]]

        has_break_points["beatmap_id"] = has_break_points["ID"].str.split("-").str[0]
        grouped = has_break_points.groupby("beatmap_id")

        results = Parallel(n_jobs=-1)(
            delayed(self.safe_process)(group)
            for _, group in tqdm(
                grouped,
                desc="Processing Groups",
            )
        )
        results = [res for res in results if res is not None]
        result = pd.concat(results)

        output_file = os.path.join(self.dataset_path, "breaks.csv")
        result.to_csv(output_file, index=False)
        print("Data saved to ", output_file)

    def merge_breaks(self):
        hit_objects_path = os.path.join(self.dataset_path, "hit_objects_formatted.csv")
        breaks_path = os.path.join(self.dataset_path, "breaks.csv")

        hit_objects_df = pd.read_csv(hit_objects_path)
        breaks_df = pd.read_csv(breaks_path)
        last_id = hit_objects_df["unique_id"].max()
        breaks_df["unique_id"] = range(last_id + 1, last_id + 1 + len(breaks_df))
        breaks_df["Type"] = "break"
        breaks_df["beatmap_id"] = breaks_df["ID"].str.split("-").str[0]

        for col in hit_objects_df.columns:
            if col not in breaks_df.columns:
                breaks_df[col] = -1
        merged_df = pd.concat([hit_objects_df, breaks_df], ignore_index=True)
        merged_df = merged_df.sort_values(by="ID").reset_index(drop=True)
        merged_df.to_csv(hit_objects_path, index=False)
        print(
            "breaks.csv merged with hit_objects_formatted.csv and saved as",
            hit_objects_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Format dataset")
    parser.add_argument(
        "--dataset_path",
        required=True,
    )
    parser.add_argument("--merge", action="store_true", default=False)
    args = parser.parse_args()

    formatter = Formatter(args.dataset_path)

    formatter.merge_breaks() if args.merge else formatter.format_dataset()


if __name__ == "__main__":
    main()
