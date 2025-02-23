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
        """Initialize dataset files with headers if they don't exist."""
        for file, headers in [
            (
                self.beatmaps_file,
                [
                    "BeatmapID",
                    "Title",
                    "Artist",
                    "Creator",
                    "Version",
                    "HPDrainRate",
                    "CircleSize",
                    "OverallDifficulty",
                    "ApproachRate",
                    "SliderMultiplier",
                    "SliderTickRate",
                ],
            ),
            (
                self.hit_objects_file,
                ["BeatmapID", "Time", "Type", "X", "Y", "HitSound", "Extra"],
            ),
            (
                self.timing_points_file,
                ["BeatmapID", "Offset", "BPM", "SV_Multiplier", "Inherited"],
            ),
        ]:
            if not os.path.exists(file):
                with open(file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)

    def write(self, data):
        hit_objects = data["hit_objects"]
        timing_points = data["timing_points"]
        metadata = data["metadata"]
        difficulty = data["difficulty"]
        beatmap_id = metadata.get("BeatmapID", None)
        if beatmap_id is None:
            print("BeatmapID missing, skipping entry.")
            return

        if self.beatmap_exists(beatmap_id):
            print(f"Beatmap {beatmap_id} already exists in beatmaps.csv, skipping.")
            return

        self.save_beatmap(beatmap_id, metadata, difficulty)
        self.save_hit_objects(beatmap_id, hit_objects)
        self.save_timing_points(beatmap_id, timing_points)

    def beatmap_exists(self, beatmap_id):
        csv_file = "dataset_csv/beatmaps.csv"

        if not os.path.exists(csv_file):
            return False

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == str(beatmap_id):
                    return True
        return False

    def save_beatmap(self, beatmap_id, metadata, difficulty):
        with open(self.beatmaps_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    beatmap_id,
                    metadata.get("Title", ""),
                    metadata.get("Artist", ""),
                    metadata.get("Creator", ""),
                    metadata.get("Version", ""),
                    difficulty.get("HPDrainRate", ""),
                    difficulty.get("CircleSize", ""),
                    difficulty.get("OverallDifficulty", ""),
                    difficulty.get("ApproachRate", ""),
                    difficulty.get("SliderMultiplier", ""),
                    difficulty.get("SliderTickRate", ""),
                ]
            )

    def save_hit_objects(self, beatmap_id, hit_objects):
        with open(self.hit_objects_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for obj in hit_objects:
                extra = obj.get("path", obj.get("end_time", ""))
                writer.writerow(
                    [
                        beatmap_id,
                        obj["time"],
                        obj["type"],
                        obj["x"],
                        obj["y"],
                        obj["hit_sound"],
                        extra,
                    ]
                )

    def save_timing_points(self, beatmap_id, timing_points):
        with open(self.timing_points_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for tp in timing_points:
                bpm = tp.get("bpm", "")
                sv = tp.get("sv_multiplier", "")
                writer.writerow([beatmap_id, tp["offset"], bpm, sv, tp["inherited"]])
