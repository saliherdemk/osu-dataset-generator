import os
import argparse
import pandas as pd
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
            "MFCC",
            "RMS",
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

    def get_timing_attributes(self, row):
        beatmap_id = row["ID"]
        target_time = row["Time"]
        beatmaps_df: pd.DataFrame = self.beatmaps_df
        time_points_df: pd.DataFrame = self.time_points_df
        base_velocity = float(
            beatmaps_df.loc[beatmaps_df["ID"] == beatmap_id, "SliderMultiplier"].iloc[0]
        )
        beatmap_time_points = time_points_df[time_points_df["ID"] == beatmap_id]

        df_filtered = beatmap_time_points[beatmap_time_points["time"] <= target_time]

        latest_uninherited = df_filtered[df_filtered["uninherited"] == 1.0]
        if not latest_uninherited.empty:
            latest_uninherited = latest_uninherited.loc[
                latest_uninherited["time"].idxmax()
            ]
        else:
            first_uninherited = beatmap_time_points[
                beatmap_time_points["uninherited"] == 1.0
            ]
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

        return {
            "beat_length": beat_length,
            "meter": meter,
            "slider_velocity": slider_velocity,
            "sample_set": sample_set,
            "volume": volume,
            "effects": effects,
        }

    def process(self, series):
        audio_folder = os.path.join(self.audio_path, str(series["beatmap_id"].iloc[0]))
        audio_file = os.path.join(audio_folder, os.listdir(audio_folder)[0])
        y, sr = librosa.load(audio_file)

        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        rms_energy = librosa.feature.rms(y=y)

        aligned_objects = []
        matched_mfccs = []
        matched_rms = []
        for time in series["Time_sec"]:
            closest_beat = min(beat_times, key=lambda x: abs(x - time))
            aligned_objects.append(closest_beat)

            frame = librosa.time_to_frames(time, sr=sr)
            matched_mfccs.append(mfcc[:, frame])
            matched_rms.append(rms_energy[0][frame])
        series["Aligned_Time"] = aligned_objects
        series["MFCC"] = matched_mfccs
        series["RMS"] = matched_rms

        timing_list = Parallel(n_jobs=-1)(
            delayed(self.get_timing_attributes)(row) for _, row in series.iterrows()
        )
        timing_df = pd.DataFrame(timing_list).apply(pd.Series)

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
            print(f"Error processing group: {set(group["ID"])}, Error: {e}")
            return None

    def format_dataset(self):
        hit_objects_df = self.hit_objects_df
        not_processed = hit_objects_df[hit_objects_df["effects"].isna()].copy()
        not_processed["Time_sec"] = not_processed["Time"] / 1000

        grouped = list(not_processed.groupby("ID"))

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
    corrupted = df["MFCC"].isna()
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
