import os
import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import librosa

final_columns = [
    "id",
    "time",
    "type",
    "x",
    "y",
    "hit_sound",
    "path",
    "repeat",
    "slider_time",
    "spinner_time",
    "new_combo",
    "beatmap_id",
    "beat_length",
    "meter",
    "slider_velocity",
    "sample_set",
    "volume",
    "effects",
    "difficulty_rating",
    "frame_time",
    "rms",
    "has_hit_object",
] + [f"mfcc_{i}" for i in range(1, 21)]


class Formatter:
    def __init__(self, dataset_path) -> None:
        beatmap_path = os.path.join(dataset_path, "beatmaps.csv")
        time_points_path = os.path.join(dataset_path, "timing_points.csv")
        hit_objects_path = os.path.join(dataset_path, "hit_objects.csv")
        self.beatmaps_df = pd.read_csv(beatmap_path)
        self.time_points_df = pd.read_csv(time_points_path)
        self.hit_objects_df = pd.read_csv(hit_objects_path)

        self.checkpoint_file = self.get_checkpoint_file(dataset_path)

        self.audio_path = os.path.join(dataset_path, "audio")

    def get_checkpoint_file(self, dataset_path):
        checkpoint_file = os.path.join(dataset_path, "formatted.csv")

        if not os.path.exists(checkpoint_file):
            df_template = pd.DataFrame(columns=final_columns)
            df_template.to_csv(checkpoint_file, index=False)

        return checkpoint_file

    def get_timing_attributes(self, series):
        beatmap_ids = series["id"].values
        target_times = series["time"].values

        selected_columns = (
            self.beatmaps_df.set_index("id")
            .loc[beatmap_ids, ["slider_multiplier", "difficulty_rating"]]
            .astype(float)
        )

        base_velocities = selected_columns["slider_multiplier"].values
        difficulty_ratings = selected_columns["difficulty_rating"].values

        grouped_time_points = self.time_points_df.groupby("id")

        results = []
        for beatmap_id, target_time, base_velocity, difficulty_rating in zip(
            beatmap_ids, target_times, base_velocities, difficulty_ratings
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
            slider_velocity = base_velocity * (
                -100 / latest_inherited["beat_length"]
                if latest_inherited is not None
                else 1
            )

            slider_velocity = round(slider_velocity, 4)

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
                }
            )

        return pd.DataFrame(results)

    def process_song(self, song_id, song_path):
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

        beatmaps.drop(columns="length", inplace=True)

        y, sr = librosa.load(song_path)  # sr=22050
        hop_length = 256
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)

        mfcc_length = mfcc.shape[0]

        all_rows = []

        for seq_id, group in beatmaps.groupby("id"):
            group = group.copy()

            n_frames = mfcc.shape[1]

            frame_times = librosa.frames_to_time(
                np.arange(n_frames), sr=sr, hop_length=hop_length
            )

            time_array = group["time"] / 1000
            frame_indices = []
            for time in time_array:
                distances = np.abs(frame_times - time)
                closest_frame_index = np.argmin(distances)
                frame_indices.append(closest_frame_index)

            group = group.reset_index(drop=True)
            group["frame_time"] = frame_times[frame_indices] * 1000
            group["mfcc"] = [mfcc[:, f] for f in frame_indices]
            group["rms"] = [rms[0][f] for f in frame_indices]
            group["has_hit_object"] = True

            for i in range(mfcc_length):
                group[f"mfcc_{i+1}"] = group["mfcc"].apply(lambda x: x[i])
            group.drop(columns="mfcc", inplace=True)

            missing_frames = [i for i in range(mfcc.shape[1]) if i not in frame_indices]

            silent_rows = []
            for f in missing_frames:
                row = {
                    "id": seq_id,
                    "type": "slient",
                    "new_combo": False,
                    "beatmap_id": int(song_id),
                    "time": librosa.frames_to_time(f, sr=sr, hop_length=hop_length)
                    * 1000,
                    "frame_time": librosa.frames_to_time(
                        f, sr=sr, hop_length=hop_length
                    )
                    * 1000,
                    "rms": rms[0][f],
                    "has_hit_object": False,
                }
                for i in range(mfcc_length):
                    row[f"mfcc_{i+1}"] = mfcc[i, f]
                silent_rows.append(row)

            silent_df = pd.DataFrame(silent_rows)
            all_rows.append(pd.concat([group, silent_df], ignore_index=True))

        final_df = pd.concat(all_rows, ignore_index=True)
        final_df = final_df.sort_values(["id", "time"]).reset_index(drop=True)

        final_df = final_df[final_columns]
        return final_df

    def format_dataset(self):
        unique_ids = set()
        chunks = pd.read_csv(
            self.checkpoint_file, usecols=["beatmap_id"], chunksize=500000
        )

        for chunk in chunks:
            unique_ids.update(str(id) for id in chunk["beatmap_id"].unique())
            del chunk

        # with open("/mnt/L-HDD/Downloads/processed.txt", "r") as f:
        #     unique_ids = [line.strip() for line in f]
        # unique_ids = set(unique_ids)

        songs_ids = os.listdir(self.audio_path)
        songs_full_paths = {
            song_id: os.path.join(
                self.audio_path,
                song_id,
                os.listdir(os.path.join(self.audio_path, song_id))[0],
            )
            for song_id in songs_ids
            if song_id not in unique_ids
        }

        # for song_id, song_path in tqdm(songs_full_paths.items()):
        #     try:
        #         df = self.process_song(song_id, song_path)
        #         df.to_csv(self.checkpoint_file, mode="a", header=False, index=False)
        #     except Exception as e:
        #         print(e)

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
