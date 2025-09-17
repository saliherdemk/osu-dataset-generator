import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def correct_effect_value(x):
    if x > 8:
        return x != 5746
    return 0 if x == 8 else x


def parse_path(path):
    p = ["<start_path>"]
    splitted = path.split("|")
    p.append(splitted.pop(0))
    for i in splitted:
        x, y = i.split(":")
        x = max(0, min(512, round(int(x) / 32) * 32))
        y = max(0, min(384, round(int(y) / 32) * 32))

        p += [f"x_{x}", f"y_{y}"]
    p.append("<end_path>")

    return ",".join(p)


def get_delta_time(dt):
    result = ["<start_delta_time>"]
    m = dt // 2000
    n = dt % 2000
    for _ in range(m):
        result.append("dt_2000")
    result += ["dt_" + str(n), "<end_delta_time>"]
    return ",".join(result)


def get_duration(t):
    result = ["<start_duration>"]
    m = t // 2000
    n = t % 2000
    for _ in range(m):
        result.append("duration_2000")
    result += ["duration_" + str(n), "<end_duration>"]
    return ",".join(result)


def get_repeat(r):
    result = ["<start_repeat>"]
    m = r // 30
    n = r % 30
    for _ in range(m):
        result.append("repeat_30")
    result += ["repeat_" + str(n), "<end_repeat>"]
    return ",".join(result)


def encode(beatmap):
    encoded = []

    for _, row in beatmap.iterrows():
        hit_obj_type = row["type"]
        hit_obj = ["<hit_object_start>", str(int(row["time"]))]

        t = "type_" + row["type"]
        delta_time = get_delta_time(row["delta_time"])

        repeat = "<start_repeat>,repeat_0,<end_repeat>"
        slider_velocity = "sv_0.0"
        duration = "<start_duration>,duration_0,<end_duration>"

        if hit_obj_type == "slider":
            # path = parse_path(row["path"])
            repeat = get_repeat(row["repeat"])
            slider_velocity = f"sv_{np.round(row["slider_velocity"], 1)}"

        if hit_obj_type != "circle":
            duration = get_duration(row["duration"])

        hit_obj += [t, delta_time, repeat, slider_velocity, duration]

        hit_obj.append("<hit_object_end>")
        encoded.append(",".join(hit_obj))
    return ",".join(encoded)


def beatmap_encoding(key, group):
    dataset = []
    encoded = "<beatmap_start>," + encode(group) + ",<beatmap_end>"
    dataset.append(
        {
            "beatmap_id": key,
            "tokenized": encoded,
        }
    )

    return pd.DataFrame(dataset)


def tokens_to_ids(text, tok_to_id):
    ids = [str(tok_to_id[token]) for token in text.split(",") if len(token)]
    return ",".join(ids)


def process(input_file, output_file):
    df = pd.read_csv(input_file)

    grouped = df.groupby("id")

    dataset = []

    for key, group in tqdm(grouped, desc="Tokenize beatmaps"):
        dataset.append(beatmap_encoding(key, group))

    dataset = pd.concat(dataset, ignore_index=True)

    dataset.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    process(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
