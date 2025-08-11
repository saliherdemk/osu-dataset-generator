import argparse

import pandas as pd


def parse_path(path):
    p = ["<start_path>"]
    splitted = path.split("|")
    p.append(splitted.pop(0))
    for i in splitted:
        x, y = i.split(":")
        p += ["x_" + x, "y_" + y]
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


def get_tick(t):
    result = ["<start_tick>"]
    m = t // 50
    n = t % 50
    for _ in range(m):
        result.append("tick_50")
    result += ["tick_" + str(n), "<end_tick>"]
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
    encoded = ["<beatmap_start>"]

    for _, row in beatmap.iterrows():
        hit_obj_type = row["type"]
        hit_obj = ["<hit_object_start>"]

        t = "type_" + row["type"]
        x = f"x_{round(row["x"] / 32) * 32}"
        y = f"y_{round(row["y"] / 32) * 32}"
        hit_sound = f"hit_sound_{row["hit_sound"]}"
        new_combo = f"new_combo_{int(row["new_combo"])}"
        sample_set = f"sample_set_{row["sample_set"]}"
        volume = f"vol_{round(row["volume"] / 10) * 10}"
        effects = f"effects_{row["effects"]}"
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
            hit_obj.append(get_tick(row["tick"]))

        hit_obj.append("<hit_object_end>")
        encoded.append(",".join(hit_obj))
    encoded.append("<beatmap_end>")
    return "".join(encoded)


def process(input_file, output_file):
    df = pd.read_csv(input_file)

    grouped = df.groupby("id")

    dataset = []
    for key, df in grouped:
        dataset.append({"beatmap_id": key, "encoded": encode(df)})

    new_df = pd.DataFrame(dataset)
    new_df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    process(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
