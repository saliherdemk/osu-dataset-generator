import csv
import os


class DataExporter:
    def __init__(self, dataset_folder):
        os.makedirs(dataset_folder, exist_ok=True)
        self.beatmaps_file = os.path.join(dataset_folder, "beatmaps.csv")
        self.hit_objects_file = os.path.join(dataset_folder, "hit_objects.csv")
        self.timing_points_file = os.path.join(dataset_folder, "timing_points.csv")

        self.init_csv()

    def init_csv(self):
        for file, headers in [
            (
                self.beatmaps_file,
                [
                    "id",
                    "title",
                    "artist",
                    "creator",
                    "version",
                    "hp_drain_rate",
                    "circle_size",
                    "overall_difficulty",
                    "approach_rate",
                    "slider_multiplier",
                    "slider_tick_rate",
                    "break_points",
                ],
            ),
            (
                self.hit_objects_file,
                [
                    "id",
                    "time",
                    "type",
                    "x",
                    "y",
                    "hit_sound",
                    "path",
                    "repeat",
                    "length",
                    "spinner_time",
                ],
            ),
            (
                self.timing_points_file,
                [
                    "id",
                    "time",
                    "beat_length",
                    "meter",
                    "sample_set",
                    "volume",
                    "uninherited",
                    "effects",
                ],
            ),
        ]:
            if not os.path.exists(file):
                with open(file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)

    def write_data(self, data, id):
        hit_objects = data["hit_objects"]
        timing_points = data["timing_points"]
        metadata = data["metadata"]
        difficulty = data["difficulty"]
        break_points = data["break_points"]

        self.save_beatmap(id, metadata, difficulty, break_points)
        self.save_hit_objects(id, hit_objects)
        self.save_timing_points(id, timing_points)

    def save_beatmap(self, id, metadata, difficulty, break_points):
        with open(self.beatmaps_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    id,
                    metadata.get("Title", ""),
                    metadata.get("Artist", ""),
                    metadata.get("Creator", ""),
                    str(metadata.get("Version", "")),
                    difficulty.get("HPDrainRate", ""),
                    difficulty.get("CircleSize", ""),
                    difficulty.get("OverallDifficulty", ""),
                    difficulty.get("ApproachRate", ""),
                    difficulty.get("SliderMultiplier", ""),
                    difficulty.get("SliderTickRate", ""),
                    break_points,
                ]
            )

    def save_hit_objects(self, id, hit_objects):
        with open(self.hit_objects_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for obj in hit_objects:
                writer.writerow(
                    [
                        id,
                        obj["time"],
                        obj["type"],
                        obj["x"],
                        obj["y"],
                        obj["hit_sound"],
                        obj["path"],
                        obj["repeat"],
                        obj["length"],
                        obj["spinner_time"],
                    ]
                )

    def save_timing_points(self, id, timing_points):
        with open(self.timing_points_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for tp in timing_points:
                writer.writerow(
                    [
                        id,
                        tp.get("time"),
                        tp.get("beat_length"),
                        tp.get("meter"),
                        tp.get("sample_set"),
                        tp.get("volume"),
                        tp.get("uninherited"),
                        tp.get("effects"),
                    ]
                )
