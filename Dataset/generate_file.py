import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv


def get_uninherited_points(rows):
    points = []

    for _, row in rows.iterrows():
        beat_length, meter, _, volume, effects, sample_set, time = row
        time_point = f"{round(time)},{beat_length},{round(meter)},{sample_set},0,{int(volume)},1,{int(effects)}"
        points.append(time_point)
    return points


def get_inherited_points(rows):
    points = []
    for _, row in rows.iterrows():
        _, meter, slider_velocity, volume, effects, sample_set, time = row
        time_point = f"{round(time)},{-100 / slider_velocity},{round(meter)},{sample_set},0,{int(volume)},0,{int(effects)}"
        points.append(time_point)
    return points


def prepare(df):
    df = df[[col for col in df.columns if not col.startswith(("diff", "mfcc"))]]
    first_true_idx = df.index[~df["type_slient"]][0]
    last_true_idx = df.index[~df["type_slient"]][-1]
    filtered = df.loc[first_true_idx:last_true_idx]

    filtered["slider_length"] = (
        (filtered["slider_time"] / filtered["beat_length"])
        * filtered["slider_velocity"]
        * 100
    )
    filtered["effects"] = df.apply(lambda row: row["effects_1.0"], axis=1)
    sample_set_cols = [
        "sample_set_0.0",
        "sample_set_1.0",
        "sample_set_2.0",
        "sample_set_3.0",
    ]
    filtered["sample_set"] = (
        filtered[sample_set_cols].idxmax(axis=1).str.extract("(\d+)")
    )
    return filtered


def get_timing_points(df):
    time_points_cols = [
        "beat_length",
        "meter",
        "slider_velocity",
        "volume",
        "effects",
        "sample_set",
        "time",
    ]

    time_filtered = df[~df["type_slient"]][time_points_cols]
    index_changes = {}
    for col in time_points_cols:
        index_changes[col] = time_filtered.index[
            time_filtered[col] != time_filtered[col].shift()
        ].tolist()

    beat_length, meter, slider_velocity, volume, effects, sample_set, _time = (
        index_changes.values()
    )

    beat_rows = time_filtered.loc[beat_length]
    meter_rows = time_filtered.loc[meter]
    slider_velocity_rows = time_filtered.loc[slider_velocity]
    volume_rows = time_filtered.loc[volume]
    effects_rows = time_filtered.loc[effects]
    sample_set_rows = time_filtered.loc[sample_set]

    all_points = []

    timing_points = get_uninherited_points(beat_rows)
    meter_points = get_uninherited_points(meter_rows)
    slider_velocity_points = get_inherited_points(slider_velocity_rows)
    volume_points = get_inherited_points(volume_rows)
    effects_points = get_inherited_points(effects_rows)
    sample_set_points = get_inherited_points(sample_set_rows)

    all_points = list(
        set(
            timing_points
            + meter_points
            + slider_velocity_points
            + volume_points
            + effects_points
            + sample_set_points
        )
    )

    return sorted(all_points, key=lambda x: int(x.split(",")[0]))


def parse(x):
    return x.idxmax(axis=1).str.split("_").str[-1]


def hit_objects_prepare(df):
    hit_filtered = df[~df["type_slient"]]

    hit_sound_cols = [col for col in df.columns if col.startswith("hit_sound_")]
    curve_type_cols = [col for col in df.columns if col.startswith("curve_type_")]
    type_cols = [col for col in df.columns if col.startswith("type_")]
    combo_cols = ["new_combo_False", "new_combo_True"]
    path_x_columns = [
        col
        for col in df.columns
        if col.startswith("path_") and int(col.split("_")[1]) % 2 != 0
    ]
    path_y_columns = [
        col
        for col in df.columns
        if col.startswith("path_") and int(col.split("_")[1]) % 2 == 0
    ]
    hit_filtered["hit_sound"] = (
        parse(hit_filtered[hit_sound_cols]).astype(float).astype(int)
    )
    hit_filtered["curve_type"] = parse(hit_filtered[curve_type_cols])
    hit_filtered["type"] = parse(hit_filtered[type_cols])
    hit_filtered["new_combo"] = parse(hit_filtered[combo_cols])

    for i, (col_x, col_y) in enumerate(zip(path_x_columns, path_y_columns)):
        hit_filtered[f"point_{i}"] = (
            hit_filtered[col_x].round(0).astype(int).astype(str)
            + ":"
            + hit_filtered[col_y].round(0).astype(int).astype(str)
        )

    hit_columns = [
        "x",
        "y",
        "time",
        "type",
        "hit_sound",
        "curve_type",
        "new_combo",
        "repeat",
        "slider_length",
        "spinner_time",
    ] + [col for col in hit_filtered.columns if col.startswith("point")]

    return hit_filtered[hit_columns]


def rounded_str(x):
    return str(int(x))


def remove_trailing_zeros(data):
    while data and data[-1] == "0:0":
        data.pop()
    return data


def get_type_params(
    hit_type, new_combo, curve_type, curve_columns, repeat, slider_length, spinner_time
):
    flag = 0
    objectParams = ""

    if hit_type == "circle":
        flag |= 1 << 0
    elif hit_type == "slider":
        flag |= 1 << 1
        curve_columns = remove_trailing_zeros(curve_columns)
        objectParams += f",{curve_type}|{"|".join(curve_columns)},{rounded_str(repeat)},{str(round(slider_length, 12))}"
    elif hit_type == "spinner":
        flag |= 1 << 3
        objectParams += "," + rounded_str(spinner_time)

    if new_combo:
        flag |= 1 << 2

    return flag, objectParams


def get_hit_objects(df):
    hit_objects = []

    for _i, row in df.iterrows():
        (
            x,
            y,
            time,
            hit_type,
            hit_sound,
            curve_type,
            new_combo,
            repeat,
            slider_length,
            spinner_time,
            *curve_columns,
        ) = row
        flag, objectParams = get_type_params(
            hit_type,
            new_combo,
            curve_type,
            curve_columns,
            repeat,
            slider_length,
            spinner_time,
        )
        hit_objects.append(
            f"{rounded_str(x)},{rounded_str(y)},{rounded_str(time)},{str(flag)},{rounded_str(hit_sound)}{str(objectParams)}"
        )
    return hit_objects


def save_file(timing_points, hit_objects, output_file):
    header = "osu file format v14\n\n[General]\nAudioFilename: audio.mp3\n\n[Difficulty]\nSliderMultiplier:1\nSliderTickRate:2\n\n"

    with open(output_file, "w") as f:
        f.write(header)
        f.write("[TimingPoints]\n")
        for line in timing_points:
            f.write(f"{line}\n")

        f.write("\n[HitObjects]\n")
        for line in hit_objects:
            f.write(f"{line}\n")


def generate_file(input_file, output_file):
    df = pd.read_csv(input_file)

    df = prepare(df)
    timing_points = get_timing_points(df)

    hit_objects_df = hit_objects_prepare(df)
    hit_objects = get_hit_objects(hit_objects_df)

    save_file(timing_points, hit_objects, output_file)


def main():
    parser = argparse.ArgumentParser(description="Split curve points")
    parser.add_argument(
        "--input_file",
        required=True,
    )

    parser.add_argument(
        "--output_file",
        required=True,
    )

    args = parser.parse_args()

    generate_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
