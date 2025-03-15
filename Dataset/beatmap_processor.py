import os


class BeatmapProcessor:
    def __init__(self, beatmapset_folder, osu_file):
        self.beatmapset_folder = beatmapset_folder
        self.osu_file = osu_file
        self.is_mode_osu = self.verify_mode()
        if self.is_mode_osu:
            self.hit_objects = self.parse_hit_objects()
            self.timing_points = self.parse_timing_points()
            self.metadata = self.parse_metadata()
            self.difficulty = self.parse_difficulty()
            self.break_points = self.parse_break_points()

    def verify_mode(self):
        with open(
            os.path.join(self.beatmapset_folder, self.osu_file), "r", encoding="utf-8"
        ) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Mode"):
                    return int(line.split(":")[1]) == 0

    def parse_hit_objects(self):
        hit_objects = []
        with open(
            os.path.join(self.beatmapset_folder, self.osu_file), "r", encoding="utf-8"
        ) as f:
            lines = f.readlines()

        in_hit_objects = False
        for line in lines:
            line = line.strip()
            if line == "[HitObjects]":
                in_hit_objects = True
                continue

            if in_hit_objects and line:
                parts = line.split(",")
                x, y, time, obj_type, hit_sound = map(int, parts[:5])
                object_data = parts[5:]

                if obj_type & 1:
                    hit_objects.append(
                        {
                            "type": "circle",
                            "x": x,
                            "y": y,
                            "time": time,
                            "hit_sound": hit_sound,
                        }
                    )

                elif obj_type & 2:
                    slider_path = object_data[0]
                    repeat = int(object_data[1])
                    hit_objects.append(
                        {
                            "type": "slider",
                            "x": x,
                            "y": y,
                            "time": time,
                            "hit_sound": hit_sound,
                            "path": slider_path,
                            "repeat": repeat,
                        }
                    )

                elif obj_type & 8:
                    end_time = int(object_data[0])
                    hit_objects.append(
                        {
                            "type": "spinner",
                            "x": x,
                            "y": y,
                            "time": time,
                            "hit_sound": hit_sound,
                            "end_time": end_time,
                        }
                    )

        return hit_objects

    def parse_timing_points(self):
        timing_points = []
        with open(
            os.path.join(self.beatmapset_folder, self.osu_file), "r", encoding="utf-8"
        ) as f:
            lines = f.readlines()

        in_timing_points = False
        for line in lines:
            line = line.strip()
            if line.startswith("["):
                in_timing_points = line == "[TimingPoints]"
                continue
            if in_timing_points and line:
                parts = line.split(",")

                while len(parts) < 8:
                    parts.append("")

                timing_points.append(
                    {
                        "time": parts[0],
                        "beat_length": parts[1],
                        "meter": parts[2],
                        "sample_set": parts[3],
                        "volume": parts[5],
                        "uninherited": parts[6],
                        "effects": parts[7],
                    }
                )

        return timing_points

    def parse_metadata(self):
        metadata = {}
        with open(
            os.path.join(self.beatmapset_folder, self.osu_file), "r", encoding="utf-8"
        ) as f:
            lines = f.readlines()

        in_metadata = False
        for line in lines:
            line = line.strip()
            if line.startswith("["):
                in_metadata = line == "[Metadata]"
                continue
            if in_metadata and line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

        return metadata

    def parse_difficulty(self):
        difficulty = {}
        with open(
            os.path.join(self.beatmapset_folder, self.osu_file), "r", encoding="utf-8"
        ) as f:
            lines = f.readlines()

        in_difficulty = False
        for line in lines:
            line = line.strip()
            if line.startswith("["):
                in_difficulty = line == "[Difficulty]"
                continue
            if in_difficulty and line:
                key, value = line.split(":", 1)
                difficulty[key.strip()] = value.strip()

        return difficulty

    def parse_break_points(self):
        breaks = []
        with open(
            os.path.join(self.beatmapset_folder, self.osu_file), "r", encoding="utf-8"
        ) as f:
            lines = f.readlines()

            in_events = False
            in_breaks = False
            for line in lines:
                line = line.strip()
                if line.startswith("["):
                    in_events = line == "[Events]"
                    continue
                if in_events and line:
                    if line.startswith("//"):
                        in_breaks = line == "//Break Periods"
                        continue
                    if in_breaks:
                        breaks.append(line)

            return breaks

    def get_data(self):
        return {
            "hit_objects": self.hit_objects,
            "timing_points": self.timing_points,
            "metadata": self.metadata,
            "difficulty": self.difficulty,
            "break_points": self.break_points,
        }
