import os
import argparse
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import librosa


class Formatter:
    def __init__(self, dataset_path) -> None:
        self.beatmaps_df = pd.read_csv(os.path.join(dataset_path, "beatmaps.csv"))
        self.audio_path = os.path.join(dataset_path, "audio")
        self.output_file = os.path.join(dataset_path, "breaks.csv")

    def process(self, series):
        print(series)
        return
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
        has_break_points = self.beatmaps_df[self.beatmaps_df["BreakPoints"] != "[]"]
        has_break_points = has_break_points[["ID", "BreakPoints"]]

        has_break_points["beatmap_id"] = has_break_points["ID"].str.split("-").str[0]
        grouped = has_break_points.groupby("beatmap_id")

        start_index = 0
        chunk_size = 100
        while start_index < len(has_break_points):
            end_index = start_index + chunk_size
            end_index = min(end_index, len(has_break_points))
            results = Parallel(n_jobs=-1)(
                delayed(self.safe_process)(group)
                for _, group in tqdm(
                    grouped[start_index:end_index],
                    desc="Processing Groups",
                )
            )
            break


def main():
    parser = argparse.ArgumentParser(description="Format dataset")
    parser.add_argument(
        "--dataset_path",
        required=True,
    )
    args = parser.parse_args()

    formatter = Formatter(args.dataset_path)
    formatter.format_dataset()


if __name__ == "__main__":
    main()
