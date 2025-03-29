import os
import argparse
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import librosa


class Formatter:
    def __init__(self, dataset_path) -> None:
        beatmap_path = os.path.join(dataset_path, "beatmaps.csv")
        self.beatmaps_df = pd.read_csv(beatmap_path, parse_dates=["ranked_date"])
        self.time_points_df = pd.read_csv(
            os.path.join(dataset_path, "timing_points.csv")
        )
        self.checkpoint_file = os.path.join(dataset_path, "hit_objects_formatted.csv")
        self.hit_objects_df = self.get_checkpoint(dataset_path)

        self.audio_path = os.path.join(dataset_path, "audio")

    def get_checkpoint(self, dataset_path):
        if os.path.exists(self.checkpoint_file):
            return pd.read_csv(self.checkpoint_file)
        df = pd.read_csv(os.path.join(dataset_path, "hit_objects.csv"))
        cols = [
            "mfcc",
            "rms",
            "beat_length",
            "meter",
            "slider_velocity",
            "sample_set",
            "volume",
            "effects",
        ]
        for col in cols:
            df[col] = pd.NA
        df["unique_id"] = range(1, len(df) + 1)
        return df

    def get_timing_attributes(self, series):
        beatmap_ids = series["id"].values
        target_times = series["time"].values
        beatmaps_df = self.beatmaps_df
        time_points_df = self.time_points_df

        base_velocities = (
            beatmaps_df.set_index("id")
            .loc[beatmap_ids, "slider_multiplier"]
            .values.astype(float)
        )

        grouped_time_points = time_points_df.groupby("id")

        results = []
        for beatmap_id, target_time, base_velocity in zip(
            beatmap_ids, target_times, base_velocities
        ):
            beatmap_time_points = grouped_time_points.get_group(beatmap_id)
            df_filtered = beatmap_time_points[
                beatmap_time_points["time"] <= target_time
            ]

            latest_uninherited = df_filtered[df_filtered["uninherited"] == 1.0]
            if not latest_uninherited.empty:
                latest_uninherited = latest_uninherited.loc[
                    latest_uninherited["time"].idxmax()
                ]
            else:
                first_uninherited = beatmap_time_points[
                    beatmap_time_points["uninherited"] == 1.0
                ]
                if first_uninherited.empty:
                    results.append(
                        {
                            "beat_length": np.nan,
                            "meter": np.nan,
                            "slider_velocity": np.nan,
                            "sample_set": np.nan,
                            "volume": np.nan,
                            "effects": np.nan,
                        }
                    )
                    continue
                latest_uninherited = first_uninherited.iloc[0]

            latest_inherited = df_filtered[df_filtered["uninherited"] == 0.0]
            latest_inherited = (
                latest_inherited.loc[latest_inherited["time"].idxmax()]
                if not latest_inherited.empty
                else None
            )

            beat_length = latest_uninherited["beat_length"]
            meter = latest_uninherited["meter"]
            slider_velocity = (
                latest_inherited["beat_length"]
                if latest_inherited is not None
                else -100 * base_velocity
            )
            sample_set = (
                latest_inherited["sample_set"] if latest_inherited is not None else 1
            )
            volume = latest_inherited["volume"] if latest_inherited is not None else 60
            effects = latest_inherited["effects"] if latest_inherited is not None else 0

            results.append(
                {
                    "beat_length": beat_length,
                    "meter": meter,
                    "slider_velocity": slider_velocity,
                    "sample_set": sample_set,
                    "volume": volume,
                    "effects": effects,
                }
            )

        return pd.DataFrame(results)

    def process(self, series):
        audio_folder = os.path.join(self.audio_path, str(series["beatmap_id"].iloc[0]))
        audio_file = os.path.join(audio_folder, os.listdir(audio_folder)[0])
        y, sr = librosa.load(audio_file)

        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        rms_energy = librosa.feature.rms(y=y)

        time_array = series["time_sec"].values
        frame_indices = librosa.time_to_frames(time_array, sr=sr)

        series["aligned_time"] = [
            min(beat_times, key=lambda x: abs(x - t)) for t in time_array
        ]
        series["mfcc"] = [mfcc[:, f] for f in frame_indices]
        series["rms"] = [rms_energy[0][f] for f in frame_indices]

        timing_df = self.get_timing_attributes(series)

        update_indices = series.index
        series.loc[
            update_indices,
            [
                "beat_length",
                "meter",
                "slider_velocity",
                "sample_set",
                "volume",
                "effects",
            ],
        ] = timing_df.values

        return series

    def safe_process(self, group):
        try:
            return self.process(group)
        except Exception as e:
            print(f"Error processing group: {set(group["id"])}, Error: {e}")
            return None

    def format_dataset(self):
        hit_objects_df = self.hit_objects_df
        not_processed = hit_objects_df[hit_objects_df["effects"].isna()].copy()
        not_processed["time_sec"] = not_processed["time"] / 1000

        grouped = list(not_processed.groupby("id"))
        print(len(grouped), "group remaining")

        start_index = 0
        chunk_size = 1000
        while start_index < len(grouped):
            end_index = start_index + chunk_size
            end_index = min(end_index, len(grouped))
            results = Parallel(n_jobs=-1)(
                delayed(self.safe_process)(group)
                for _, group in tqdm(
                    grouped[start_index:end_index], desc="Processing Groups"
                )
            )

            results = [res for res in results if res is not None]
            result = pd.concat(results)
            self.hit_objects_df = self.hit_objects_df.set_index("unique_id")
            result = result.set_index("unique_id")

            self.hit_objects_df.update(result)

            self.hit_objects_df = self.hit_objects_df.reset_index()
            result.reset_index()

            self.hit_objects_df.to_csv(self.checkpoint_file, index=False)

            start_index = end_index


def clear(dataset_path):
    file_path = os.path.join(dataset_path, "hit_objects_formatted.csv")
    df = pd.read_csv(file_path)
    corrupted = df["mfcc"].isna()
    num_of_corrupted = len(df[corrupted])
    if num_of_corrupted > 0:
        df = df[~corrupted]
        df.to_csv(file_path, index=False)
    print(f"Removed {num_of_corrupted} rows.")


def main():
    parser = argparse.ArgumentParser(description="Format dataset")
    parser.add_argument(
        "--dataset_path",
        required=True,
    )
    parser.add_argument("--clear", action="store_true", default=False)
    args = parser.parse_args()

    if args.clear:
        clear(args.dataset_path)
        return

    formatter = Formatter(args.dataset_path)
    formatter.format_dataset()


if __name__ == "__main__":
    main()
