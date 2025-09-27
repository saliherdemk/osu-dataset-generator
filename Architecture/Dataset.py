import os

import librosa
import pandas as pd
from torch import chunk
from torch.utils.data import DataLoader, Dataset


class BeatmapChunkDataset(Dataset):
    def __init__(
        self, input_folder, chunk_size=10000, step_size=5000, sr=22050, n_mfcc=20
    ):
        self.audio_folder = os.path.join(input_folder, "audio")
        self.chunk_size = chunk_size
        self.step_size = step_size

        self.sr = sr
        self.n_mfcc = n_mfcc

        chunk_df = pd.read_csv(os.path.join(input_folder, "chunked.csv"))
        self.groups = list(chunk_df.groupby(["id", "chunk_id"]))

        self.audio_lengths = {}
        self.total_chunks = {}
        self._precompute_audio_info()

    def _precompute_audio_info(self):
        beatmapset_ids = set()
        for (beatmap_id, chunk_id), group in self.groups:
            beatmapset_id = beatmap_id.split("-")[0]
            beatmapset_ids.add(beatmapset_id)

        for beatmapset_id in beatmapset_ids:
            audio_path = self.get_audio_path(beatmapset_id)
            duration_ms = int(librosa.get_duration(path=audio_path) * 1000)
            self.audio_lengths[beatmapset_id] = duration_ms
            total_chunks = len(range(0, duration_ms, self.step_size))
            self.total_chunks[beatmapset_id] = total_chunks

        filtered = []
        for (beatmap_id, chunk_id), group in self.groups:
            beatmapset_id = beatmap_id.split("-")[0]
            max_chunk = self.total_chunks[beatmapset_id]
            group = group[group["chunk_id"] < max_chunk]
            if not group.empty:
                filtered.append(group)

        if filtered:
            df_filtered = pd.concat(filtered, ignore_index=True)
            self.groups = df_filtered.groupby(["id", "chunk_id"])

    def get_audio_path(self, beatmapset_id):
        for ext in (".mp3", ".ogg"):
            path = os.path.join(self.audio_folder, beatmapset_id + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No audio file found for {beatmapset_id}")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        (beatmap_id, chunk_id), group = list(self.groups)[idx]
        beatmapset_id = beatmap_id.split("-")[0]

        start_ms, end_ms = group[["chunk_start", "chunk_end"]].iloc[0]
        duration = (end_ms - start_ms) / 1000.0

        audio_path = self.get_audio_path(beatmapset_id)
        y, _ = librosa.load(
            audio_path, sr=self.sr, offset=start_ms / 1000, duration=duration
        )

        features = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc).T

        tokens = group["tokenized"].iloc[0]
        return {
            "beatmap_id": beatmap_id,
            "chunk_id": chunk_id,
            "audio": features,
            "hit_objects": tokens,
        }


def createDataLoader(input_folder, batch_size):
    dataset = BeatmapChunkDataset(input_folder)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
