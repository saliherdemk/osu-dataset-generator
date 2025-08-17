import argparse
import json
import os

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
        hit_obj = ["<hit_object_start>"]

        t = "type_" + row["type"]
        x = f"x_{max(0, min(512, round(row["x"] / 32) * 32))}"
        y = f"y_{max(0, min(384, round(row["y"] / 32) * 32))}"
        hit_sound = f"hit_sound_{row["hit_sound"]}"
        new_combo = f"new_combo_{int(row["new_combo"])}"
        sample_set = f"sample_set_{row["sample_set"]}"
        volume = f"vol_{round(row["volume"] / 10) * 10}"
        effects = f"effects_{correct_effect_value(row["effects"])}"
        delta_time = get_delta_time(row["delta_time"])

        hit_obj += [
            t,
            x,
            y,
            hit_sound,
            new_combo,
            sample_set,
            volume,
            effects,
            delta_time,
        ]

        if hit_obj_type == "slider":
            path = parse_path(row["path"])
            repeat = get_repeat(row["repeat"])
            slider_velocity = f"sv_{round(row["slider_velocity"], 1)}"
            hit_obj += [path, repeat, slider_velocity]

        if hit_obj_type != "circle":
            hit_obj.append(get_duration(row["duration"]))

        hit_obj.append("<hit_object_end>")
        encoded.append(",".join(hit_obj))
    return ",".join(encoded)


def chunk_encoding(key, group):
    sr = 22050
    hop_length = 512
    chunk_size = 512

    chunk_duration = (chunk_size * hop_length) / sr
    group["time_sec"] = group["time"] / 1000

    group["chunk_idx"] = (group["time_sec"] // chunk_duration).astype(int)

    chunked_objects = group.groupby("chunk_idx")
    dataset = []

    for chunk_idx, group in chunked_objects:
        encoded = encode(group)
        if chunk_idx == 0:
            encoded = "<beatmap_start>" + encoded
        if chunk_idx == len(chunked_objects) - 1:
            encoded = encoded + "<beatmap_end>"
        dataset.append({"beatmap_id": key, "chunk": chunk_idx, "tokenized": encoded})
    return pd.DataFrame(dataset)


def tokens_to_ids(text, tok_to_id):
    ids = [str(tok_to_id[token]) for token in text.split(",")]
    return ",".join(ids)


def process(input_file, output_file):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "vocab/token2id.json")
    with open(filename, "r") as f:
        tok_to_id = json.load(f)

    df = pd.read_csv(input_file)

    grouped = df.groupby("id")

    dataset = []

    for key, group in tqdm(grouped, desc="Tokenize beatmaps"):
        dataset.append(chunk_encoding(key, group))

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
