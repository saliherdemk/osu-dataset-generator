import os

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class BeatmapChunkDataset(Dataset):
    def __init__(self, input_folder):
        self.audio_folder = os.path.join(input_folder, "audio")

        df = pd.read_csv(os.path.join(input_folder, "chunked.csv"))
        self.groups = list(df.groupby(["id", "chunk_id"]))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        (beatmap_id, chunk_id), group = self.groups[idx]

        features = torch.tensor(
            group[
                [
                    "type_circle",
                    "type_slider",
                    "type_spinner",
                    "hit_start_rel",
                    "hit_end_rel",
                ]
            ].values,
            dtype=torch.float32,
        )
        beatmapset_id = beatmap_id.split("-")[0]
        chunk_audio_path = os.path.join(
            self.audio_folder, f"{beatmapset_id}_chunk{chunk_id}.pt"
        )

        difficulty_rating = torch.tensor(
            [group.iloc[0]["difficulty_rating"]], dtype=torch.float32
        )

        return {
            "beatmap_id": beatmap_id,
            "chunk_id": chunk_id,
            "features": features,
            "audio": torch.load(chunk_audio_path),
            "difficulty_rating": difficulty_rating,
        }


def collate_fn(batch):
    beatmap_ids = [item["beatmap_id"] for item in batch]
    chunk_ids = [item["chunk_id"] for item in batch]

    features_list = [item["features"] for item in batch]
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    tgt_key_padding_mask = torch.zeros(features_padded.shape[:2], dtype=torch.bool)
    for i, feat in enumerate(features_list):
        tgt_key_padding_mask[i, : feat.shape[0]] = 1

    max_tgt_len = features_padded.size(1)
    tgt_causal_mask = torch.triu(
        torch.ones(max_tgt_len, max_tgt_len, dtype=torch.bool), diagonal=1
    )

    audio_list = [item["audio"].squeeze(0) for item in batch]
    audio_padded = pad_sequence(audio_list, batch_first=True, padding_value=0.0)

    audio_mask = torch.zeros(audio_padded.shape[:2], dtype=torch.bool)
    for i, a in enumerate(audio_list):
        audio_mask[i, : a.shape[0]] = 1

    difficulty_ratings = torch.tensor(
        [item["difficulty_rating"] for item in batch], dtype=torch.float
    ).unsqueeze(1)

    return {
        "beatmap_ids": beatmap_ids,
        "chunk_ids": torch.tensor(chunk_ids, dtype=torch.long),
        "features": features_padded,
        "tgt_key_padding_mask": tgt_key_padding_mask,
        "tgt_causal_mask": tgt_causal_mask,
        "audio": audio_padded,
        "audio_mask": audio_mask,
        "difficulty_rating": difficulty_ratings,
    }


def createDataLoader(input_folder, batch_size):
    dataset = BeatmapChunkDataset(input_folder)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    return dataloader
