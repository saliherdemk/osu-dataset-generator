import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

COL_TYPES = {
    "id": "string",
    "time": "float64",
    "type": "string",
    "x": "int16",
    "y": "int16",
    "hit_sound": "int8",
    "path": "string",
    "repeat": "int16",
    "spinner_time": "int32",
    "new_combo": "bool",
    "slider_velocity": "float64",
    "sample_set": "int8",
    "volume": "int8",
    "effects": "int8",
    "difficulty_rating": "float16",
    "meter": "int8",
    "beat_length": "float64",
    "mapper_id": "int64",
    "beatmap_id": "int64",
    "duration": "int64",
    "delta_time": "int64",
}


class Formatter:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        self.beatmaps_df = pd.read_csv(os.path.join(dataset_path, "beatmaps.csv"))
        self.time_points_df = pd.read_csv(
            os.path.join(dataset_path, "timing_points.csv")
        )
        self.hit_objects_df = pd.read_csv(os.path.join(dataset_path, "hit_objects.csv"))

        self.mel_folder, self.checkpoint_file = self.setup_output_paths()
        self.audio_path = os.path.join(dataset_path, "audio")

    def setup_output_paths(self):
        mel_folder = os.path.join(self.dataset_path, "formatted", "mels")
        os.makedirs(mel_folder, exist_ok=True)

        checkpoint_file = os.path.join(self.dataset_path, "formatted", "formatted.csv")
        if not os.path.exists(checkpoint_file):
            pd.DataFrame(columns=COL_TYPES.keys()).to_csv(checkpoint_file, index=False)

        return mel_folder, checkpoint_file

    def extract_timing_attributes(self, group):
        beatmap_ids = group["id"].values
        target_times = group["time"].values

        selected_info = self.beatmaps_df.set_index("id").loc[beatmap_ids]
        base_velocities = selected_info["slider_multiplier"].values
        difficulty_ratings = selected_info["difficulty_rating"].values
        mapper_ids = selected_info["mapper_id"].values

        grouped_timing = self.time_points_df.groupby("id")

        results = []
        for b_id, t_time, base_vel, diff, mapper in zip(
            beatmap_ids, target_times, base_velocities, difficulty_ratings, mapper_ids
        ):
            tp_group = grouped_timing.get_group(b_id)
            relevant_tp = tp_group[tp_group["time"] <= t_time]

            uninherited_candidates = relevant_tp[relevant_tp["uninherited"] == 1.0]
            if not uninherited_candidates.empty:
                latest_uninherited = uninherited_candidates.loc[
                    uninherited_candidates["time"].idxmax()
                ]
            else:
                latest_uninherited = tp_group.iloc[0]

            inherited_candidates = relevant_tp[relevant_tp["uninherited"] == 0.0]
            if not inherited_candidates.empty:
                latest_inherited = inherited_candidates.loc[
                    inherited_candidates["time"].idxmax()
                ]
            else:
                latest_inherited = latest_uninherited.copy()
                latest_inherited["beat_length"] = -100

            beat_length = latest_uninherited["beat_length"]
            meter = latest_uninherited["meter"]
            rel_sv = max(min(10, -100 / latest_inherited["beat_length"]), 0.1)
            slider_velocity = base_vel * rel_sv
            sample_set = latest_inherited["sample_set"]
            volume = latest_inherited["volume"]
            effects = latest_inherited["effects"]

            results.append(
                {
                    "beat_length": beat_length,
                    "meter": meter,
                    "slider_velocity": slider_velocity,
                    "sample_set": sample_set,
                    "volume": volume,
                    "effects": effects,
                    "difficulty_rating": diff,
                    "mapper_id": mapper,
                }
            )
        return pd.DataFrame(results)

    def save_mel_spectrogram(self, song_id, song_path, chunk_size=512):
        y, sr = librosa.load(song_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        log_mel_spec = log_mel_spec.T

        n_chunks = int(np.ceil(log_mel_spec.shape[0] / chunk_size))
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = log_mel_spec[start:end]

            if chunk.shape[0] < chunk_size:
                pad_len = chunk_size - chunk.shape[0]
                chunk = np.pad(chunk, ((0, pad_len), (0, 0)), mode="constant")

            chunks.append(chunk)

        # np.save(os.path.join(self.mel_folder, f"{song_id}.npy"), np.array(chunks))
        for idx, c in enumerate(chunks):
            np.save(os.path.join(self.mel_folder, f"{song_id}_{idx}.npy"), c)

        print(f"Saved {len(chunks)} chunks for song {song_id}")

    def process_song(self, song_id, song_path):
        self.save_mel_spectrogram(song_id, song_path)

        beatmap_data = self.hit_objects_df[
            self.hit_objects_df["beatmap_id"] == int(song_id)
        ].copy()

        timing_data = [
            self.extract_timing_attributes(group)
            for _, group in beatmap_data.groupby("id")
        ]
        timing_df = pd.concat(timing_data, ignore_index=True)

        beatmap_data.reset_index(drop=True, inplace=True)
        beatmap_data = pd.concat([beatmap_data, timing_df], axis=1)

        def compute_duration(row):
            duration = 0
            if row["type"] == "slider":
                duration = (
                    row["length"] / (row["slider_velocity"] * 100) * row["beat_length"]
                )
            elif row["type"] == "spinner":
                duration = row["spinner_time"] - row["time"]
            return int(duration)

        beatmap_data["duration"] = beatmap_data.apply(compute_duration, axis=1)
        beatmap_data["delta_time"] = (
            beatmap_data.groupby("id")["time"].diff().fillna(0).astype(int)
        )

        beatmap_data.drop(columns="length", inplace=True)
        beatmap_data = beatmap_data[COL_TYPES.keys()].astype(COL_TYPES)

        return beatmap_data

    def format_dataset(self):
        processed_ids = self.get_already_processed_ids()

        song_paths = {
            song_id: os.path.join(
                self.audio_path,
                song_id,
                os.listdir(os.path.join(self.audio_path, song_id))[0],
            )
            for song_id in os.listdir(self.audio_path)
            if song_id not in processed_ids
        }

        for song_id, path in tqdm(
            song_paths.items(), total=len(song_paths), desc="Processing songs"
        ):
            df = self.process_song(song_id, path)
            df.to_csv(self.checkpoint_file, mode="a", header=False, index=False)

    def get_already_processed_ids(self):
        processed_ids = set()
        for chunk in pd.read_csv(
            self.checkpoint_file, usecols=["beatmap_id"], chunksize=500_000
        ):
            processed_ids.update(str(bid) for bid in chunk["beatmap_id"].unique())
        return processed_ids


def main():
    parser = argparse.ArgumentParser(
        description="Format beatmap dataset with timing and audio features."
    )
    parser.add_argument(
        "--dataset_path", required=True, help="Path to dataset root folder."
    )
    args = parser.parse_args()

    formatter = Formatter(args.dataset_path)
    formatter.format_dataset()


if __name__ == "__main__":
    main()
