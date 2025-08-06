import argparse
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

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


def get_beatmapset_metadata(beatmapset_id, access_token):
    url = f"https://osu.ppy.sh/api/v2/beatmapsets/{beatmapset_id}"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        beatmapset = response.json()
        ranked_date = beatmapset.get("ranked_date")
        status = beatmapset.get("status", "Unknown")
        mapper_id = beatmapset.get("user_id")

        beatmaps = {}
        for beatmap in beatmapset.get("beatmaps"):
            beatmaps[beatmap["version"]] = beatmap["difficulty_rating"]

        return (status, ranked_date, mapper_id, beatmaps)
    else:
        return f"Error: {response.status_code} - {response.text}"


def add_metadata(dataset_folder):

    beatmaps_csv = os.path.join(dataset_folder, "beatmaps.csv")
    beatmaps_df = pd.read_csv(beatmaps_csv, keep_default_na=False)

    beatmapset_ids = set(beatmaps_df["id"].str.split("-").str[0])

    access_token = get_access_token()

    metadatas = {}
    for beatmapset_id in tqdm(beatmapset_ids):
        metadatas[beatmapset_id] = get_beatmapset_metadata(beatmapset_id, access_token)
        time.sleep(0.2)

    def fill_metadatas(row):
        beatmapset_id = row["id"].split("-")[0]
        version = row["version"]

        (status, ranked_date, mapper_id, beatmaps) = metadatas[beatmapset_id]
        row["status"] = status
        row["ranked_date"] = ranked_date
        row["mapper_id"] = mapper_id
        row["difficulty_rating"] = beatmaps[version]
        return row

    beatmaps_df = beatmaps_df.apply(fill_metadatas, axis=1)
    beatmaps_df.to_csv(beatmaps_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Add metadata.")
    parser.add_argument(
        "--dataset_folder",
        required=True,
        help="Path to the folder containing dataset folders.",
    )

    args = parser.parse_args()

    add_metadata(args.dataset_folder)


if __name__ == "__main__":
    main()
