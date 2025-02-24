import csv
import os


class DataExporter:
    def __init__(self, dataset_folder):
        os.makedirs(dataset_folder, exist_ok=True)
        self.beatmaps_file = os.path.join(dataset_folder, "beatmaps.csv")
        self.hit_objects_file = os.path.join(dataset_folder, "hit_objects.csv")
        self.timing_points_file = os.path.join(dataset_folder, "timing_points.csv")
        self.audio_file = os.path.join(dataset_folder, "audio.csv")

        self.init_csv()

    def init_csv(self):
        for file, headers in [
            (
                self.beatmaps_file,
                [
                    "ID",
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
                ["ID", "Time", "Type", "X", "Y", "HitSound", "Extra"],
            ),
            (
                self.timing_points_file,
                [
                    "ID",
                    "offset",
                    "ms_per_beat",
                    "time_signature",
                    "meter",
                    "sample_set",
                    "sample_index",
                    "volume",
                    "effects",
                ],
            ),
            (self.audio_file, ["ID", "values", "sr", "corrupted"]),
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

        if self.beatmap_exists(id):
            print(f"Beatmap {id} already exists in beatmaps.csv, skipping.")
            return

        self.save_beatmap(id, metadata, difficulty)
        self.save_hit_objects(id, hit_objects)
        self.save_timing_points(id, timing_points)

    def beatmap_exists(self, id):
        with open(self.beatmaps_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == str(id):
                    return True
        return False

    def save_beatmap(self, id, metadata, difficulty):
        with open(self.beatmaps_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    id,
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

    def save_hit_objects(self, id, hit_objects):
        with open(self.hit_objects_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for obj in hit_objects:
                extra = obj.get("path", obj.get("end_time", ""))
                writer.writerow(
                    [
                        id,
                        obj["time"],
                        obj["type"],
                        obj["x"],
                        obj["y"],
                        obj["hit_sound"],
                        extra,
                    ]
                )

    def save_timing_points(self, id, timing_points):
        with open(self.timing_points_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for tp in timing_points:
                writer.writerow(
                    [
                        id,
                        tp.get("offset"),
                        tp.get("ms_per_beat"),
                        tp.get("time_signature"),
                        tp.get("meter"),
                        tp.get("sample_set"),
                        tp.get("sample_index"),
                        tp.get("volume"),
                        tp.get("effects"),
                    ]
                )

    def save_audio(self, id, audioData):
        (audio, not_corrupted) = audioData
        with open(self.audio_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            values, sr = audio

            values_list = values.tolist()

            writer.writerow([id, values_list, sr, not not_corrupted])
