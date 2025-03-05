import os
import argparse
from tqdm import tqdm
import pandas as pd
import shutil
import requests
from dotenv import load_dotenv
import time

load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")


def get_access_token():
    url = "https://osu.ppy.sh/oauth/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": "public",
    }

    response = requests.post(url, headers=headers, data=data)
    return response.json().get("access_token")


def get_beatmapset_status(beatmapset_id, access_token):
    url = f"https://osu.ppy.sh/api/v2/beatmapsets/{beatmapset_id}"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        beatmapset = response.json()
        ranked_date = beatmapset.get("ranked_date")
        status = beatmapset.get("status", "Unknown")
        return (status, ranked_date)
    else:
        return f"Error: {response.status_code} - {response.text}"


def filter_ranked_maps(dataset_folder):

    beatmaps_csv = os.path.join(dataset_folder, "beatmaps.csv")
    hit_objects_csv = os.path.join(dataset_folder, "hit_objects.csv")
    time_points_csv = os.path.join(dataset_folder, "timing_points.csv")

    beatmaps_df = pd.read_csv(beatmaps_csv)
    hit_objects_df = pd.read_csv(hit_objects_csv)
    time_points_df = pd.read_csv(time_points_csv)

    beatmapset_ids = set(beatmaps_df["ID"].str.split("-").str[0])

    access_token = get_access_token()
    results = {}
    for beatmapset_id in tqdm(beatmapset_ids):
        results[beatmapset_id] = get_beatmapset_status(beatmapset_id, access_token)
        time.sleep(0.2)

    ranked_beatmaps = {k: v for k, v in results.items() if v[0] == "ranked"}
    ids = {k for k, _ in ranked_beatmaps.items()}
    ranked_dates = {k: v[1] for k, v in ranked_beatmaps.items()}

    beatmaps_df["beatmap_id"] = beatmaps_df["ID"].str.split("-").str[0]
    hit_objects_df["beatmap_id"] = hit_objects_df["ID"].str.split("-").str[0]
    time_points_df["beatmap_id"] = time_points_df["ID"].str.split("-").str[0]

    ids = list(ids)

    beatmaps_df = beatmaps_df[beatmaps_df["beatmap_id"].isin(ids)]
    hit_objects_df = hit_objects_df[hit_objects_df["beatmap_id"].isin(ids)]
    time_points_df = time_points_df[time_points_df["beatmap_id"].isin(ids)]
    beatmaps_df["ranked_date"] = (
        beatmaps_df["ID"].str.split("-").str[0].map(ranked_dates)
    )

    beatmaps_df.to_csv(os.path.join(dataset_folder, "beatmaps.csv"), index=False)
    hit_objects_df.to_csv(os.path.join(dataset_folder, "hit_objects.csv"), index=False)
    time_points_df.to_csv(
        os.path.join(dataset_folder, "timing_points.csv"), index=False
    )

    audio_folder = os.path.join(dataset_folder, "audio")
    folders = os.listdir(audio_folder)
    for folder in tqdm(folders):
        if folder not in ids:
            shutil.rmtree(os.path.join(audio_folder, folder))


def main():
    parser = argparse.ArgumentParser(description="Filter ranked maps.")
    parser.add_argument(
        "--dataset_folder",
        required=True,
        help="Path to the folder containing dataset folders.",
    )

    args = parser.parse_args()

    filter_ranked_maps(args.dataset_folder)


if __name__ == "__main__":
    main()
