import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

col_types = {
    "id": "string",
    "time": "float64",
    "type": "string",
    "x": "int16",
    "y": "int16",
    "hit_sound": "int8",
    "path": "string",
    "repeat": "int16",
    "slider_time": "float64",
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
    "tick": "float64"
}


class Formatter:
    def __init__(self, dataset_path) -> None:
        beatmap_path = os.path.join(dataset_path, "beatmaps.csv")
        time_points_path = os.path.join(dataset_path, "timing_points.csv")
        hit_objects_path = os.path.join(dataset_path, "hit_objects.csv")
        self.beatmaps_df = pd.read_csv(beatmap_path)
        self.time_points_df = pd.read_csv(time_points_path)
        self.hit_objects_df = pd.read_csv(hit_objects_path)

        self.mel_folder, self.checkpoint_file = self.get_output_folder(dataset_path)
        self.audio_path = os.path.join(dataset_path, "audio")

    def get_output_folder(self, dataset_path):
        mel_folder = os.path.join(dataset_path, "formatted", "mels")
        os.makedirs(mel_folder, exist_ok=True)
        checkpoint_file = os.path.join(dataset_path, "formatted", "formatted.csv")

        if not os.path.exists(checkpoint_file):
            df_template = pd.DataFrame(columns=col_types.keys())
            df_template.to_csv(checkpoint_file, index=False)

        return mel_folder, checkpoint_file

    def get_timing_attributes(self, series):
        beatmap_ids = series["id"].values
        target_times = series["time"].values

        selected_columns = (
            self.beatmaps_df.set_index("id")
            .loc[beatmap_ids, ["slider_multiplier", "difficulty_rating", "mapper_id"]]
            .astype(float)
        )

        base_velocities = selected_columns["slider_multiplier"].values
        difficulty_ratings = selected_columns["difficulty_rating"].values
        mapper_ids = selected_columns["mapper_id"].values

        grouped_time_points = self.time_points_df.groupby("id")

        results = []
        for beatmap_id, target_time, base_velocity, difficulty_rating, mapper_id in zip(
            beatmap_ids, target_times, base_velocities, difficulty_ratings, mapper_ids
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
                latest_uninherited = first_uninherited.iloc[0]

            latest_inherited = df_filtered[df_filtered["uninherited"] == 0.0]
            latest_inherited = (
                latest_inherited.loc[latest_inherited["time"].idxmax()]
                if not latest_inherited.empty
                else None
            )

            beat_length = latest_uninherited["beat_length"]
            meter = latest_uninherited["meter"]
            slider_velocity = base_velocity * (
                -100 / latest_inherited["beat_length"]
                if latest_inherited is not None
                else 1
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
                    "difficulty_rating": difficulty_rating,
                    "mapper_id": mapper_id,
                }
            )

        return pd.DataFrame(results)

    def save_mel(self, song_id, song_path):
        y, sr = librosa.load(song_path, sr=22050)

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        np.save(os.path.join(self.mel_folder, f"{song_id}.npy"), log_mel_spec)

    def process_song(self, song_id, song_path):
        self.save_mel(song_id, song_path)
        beatmaps = self.hit_objects_df[
            self.hit_objects_df["beatmap_id"] == int(song_id)
        ].copy()

        timing_attributes = [
            self.get_timing_attributes(series) for _, series in beatmaps.groupby("id")
        ]
        timing_df = pd.concat(timing_attributes, ignore_index=True)

        beatmaps = beatmaps.reset_index(drop=True)
        beatmaps = pd.concat([beatmaps, timing_df], axis=1)

        beatmaps["slider_time"] = (
            beatmaps["length"] / (beatmaps["slider_velocity"] * 100)
        ) * beatmaps["beat_length"]

        def calculate_tick(row):
            if row["type"] == "slider":
                return (row["slider_time"] / row["beat_length"]) * row["meter"]
            elif row["type"] == "spinner":
                return ((row["spinner_time"] - row["time"]) / row["beat_length"]) * row["meter"]
            else:
                return 0 

        beatmaps["tick"] = beatmaps.apply(calculate_tick, axis=1)

        beatmaps.drop(columns="length", inplace=True)
        final_df = beatmaps[col_types.keys()]
        final_df = final_df.astype(col_types)
        return final_df

    def format_dataset(self):
        processed = set()
        chunks = pd.read_csv(
            self.checkpoint_file, usecols=["beatmap_id"], chunksize=500000
        )

        for chunk in chunks:
            processed.update(str(id) for id in chunk["beatmap_id"].unique())
            del chunk

        songs_ids = os.listdir(self.audio_path)
        songs_full_paths = {
            song_id: os.path.join(
                self.audio_path,
                song_id,
                os.listdir(os.path.join(self.audio_path, song_id))[0],
            )
            for song_id in songs_ids
            if song_id not in processed
        }

        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            futures = {
                executor.submit(self.process_song, song_id, song_path): song_id
                for song_id, song_path in songs_full_paths.items()
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                final_df = future.result()
                final_df.to_csv(
                    self.checkpoint_file, mode="a", header=False, index=False
                )


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
