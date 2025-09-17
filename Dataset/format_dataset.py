import argparse
import os

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

        self.checkpoint_file = self.setup_output_paths()
        self.seperate_beatmap_id()

        self.timing_by_id = {
            bid: group.sort_values("time")
            for bid, group in self.time_points_df.groupby("id")
        }

    def setup_output_paths(self):
        checkpoint_file = os.path.join(self.dataset_path, "formatted.csv")

        if not os.path.exists(checkpoint_file):
            pd.DataFrame(columns=COL_TYPES.keys()).to_csv(checkpoint_file, index=False)
        else:
            print("formatted.csv exists. Exiting.")
            exit()

        return checkpoint_file

    def seperate_beatmap_id(self):
        self.beatmaps_df["beatmap_id"] = (
            self.beatmaps_df["id"].str.split("-").str[0].astype("int64")
        )
        self.hit_objects_df["beatmap_id"] = (
            self.hit_objects_df["id"].str.split("-").str[0].astype("int64")
        )
        self.time_points_df["beatmap_id"] = (
            self.time_points_df["id"].str.split("-").str[0].astype("int64")
        )

    def extract_timing_attributes(self, group):
        beatmap_ids = group["id"].values
        target_times = group["time"].values

        selected_info = self.beatmaps_df.set_index("id").loc[beatmap_ids]
        base_velocities = selected_info["slider_multiplier"].values
        difficulty_ratings = selected_info["difficulty_rating"].values
        mapper_ids = selected_info["mapper_id"].values

        results = []
        for b_id, t_time, base_vel, diff, mapper in zip(
            beatmap_ids, target_times, base_velocities, difficulty_ratings, mapper_ids
        ):
            tp_group = self.timing_by_id[b_id]
            relevant_tp = tp_group[tp_group["time"] <= t_time]

            uninherited_candidates = relevant_tp[relevant_tp["uninherited"] == 1.0]
            latest_uninherited = (
                uninherited_candidates.loc[uninherited_candidates["time"].idxmax()]
                if not uninherited_candidates.empty
                else tp_group.iloc[0]
            )

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

    def process_group(self, beatmap_data):
        timing_data = [
            self.extract_timing_attributes(group)
            for _, group in beatmap_data.groupby("id")
        ]
        timing_df = pd.concat(timing_data, ignore_index=True)

        beatmap_data = pd.concat(
            [beatmap_data.reset_index(drop=True), timing_df], axis=1
        )

        beatmap_data["duration"] = 0
        mask_slider = beatmap_data["type"] == "slider"
        mask_spinner = beatmap_data["type"] == "spinner"

        beatmap_data.loc[mask_slider, "duration"] = (
            beatmap_data.loc[mask_slider, "length"]
            / (beatmap_data.loc[mask_slider, "slider_velocity"] * 100)
            * beatmap_data.loc[mask_slider, "beat_length"]
        ).astype(int)

        beatmap_data.loc[mask_spinner, "duration"] = (
            beatmap_data.loc[mask_spinner, "spinner_time"]
            - beatmap_data.loc[mask_spinner, "time"]
        ).astype(int)

        beatmap_data["delta_time"] = (
            beatmap_data.groupby("id")["time"]
            .diff()
            .fillna(beatmap_data["time"])
            .astype(int)
        )

        beatmap_data.drop(columns="length", inplace=True)
        beatmap_data = beatmap_data[COL_TYPES.keys()].astype(COL_TYPES)

        return beatmap_data

    def format_dataset(self):

        beatmap_groups = {
            bid: group for bid, group in self.hit_objects_df.groupby("beatmap_id")
        }

        all_results = []
        for _bid, group in tqdm(
            beatmap_groups.items(), total=len(beatmap_groups), desc="Processing songs"
        ):
            df = self.process_group(group)
            all_results.append(df)

        if all_results:
            pd.concat(all_results).to_csv(
                self.checkpoint_file, mode="a", header=False, index=False
            )


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
