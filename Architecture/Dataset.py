import os

import pandas as pd
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class BeatmapChunkDataset(Dataset):
    def __init__(self, input_folder):
        self.audio_folder = os.path.join(input_folder, "audio")

        chunk_df = pd.read_csv(os.path.join(input_folder, "chunked.csv"))
        self.chunk_limits = pd.read_csv(os.path.join(input_folder, "chunk_limits.csv"))
        self.groups = list(chunk_df.groupby(["id", "chunk_id"]))

    def get_audio_path(self, beatmapset_id):
        for ext in (".mp3", ".ogg"):
            path = os.path.join(self.audio_folder, beatmapset_id + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No audio file found for {beatmapset_id}")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        (beatmap_id, chunk_id), group = self.groups[idx]

        beatmapset_id = beatmap_id.split("-")[0]
        audio_path = self.get_audio_path(beatmapset_id)

        limits = self.chunk_limits[
            (self.chunk_limits["id"] == beatmap_id)
            & (self.chunk_limits["chunk_id"] == chunk_id)
        ].iloc[0]

        start_ms, end_ms = limits["start"], limits["end"]

        waveform, sr = torchaudio.load(audio_path)

        start_frame = int(sr * (start_ms / 1000))

        is_last_chunk = (
            chunk_id
            == self.chunk_limits[self.chunk_limits["id"] == beatmap_id][
                "chunk_id"
            ].max()
        )

        if is_last_chunk:
            waveform = waveform[:, start_frame:]
        else:
            end_frame = int(sr * (end_ms / 1000))
            waveform = waveform[:, start_frame:end_frame]

        torchaudio.save(
            f"/home/saliherdemk/try_dataset/chunked/{beatmap_id}_{chunk_id}.wav",
            waveform,
            sr,
        )

        return {
            "id": beatmap_id,
            "chunk_id": chunk_id,
            "audio": waveform,
            "sample_rate": sr,
            "hit_objects": group,
        }


def collate_fn(batch):
    # batch is a list of dicts from __getitem__
    audios = [item["audio"].squeeze(0).T for item in batch]
    # -> shape: (time, channels) instead of (channels, time)
    # assuming mono audio, so squeeze(0) is safe

    lengths = [audio.shape[0] for audio in audios]  # keep track of true lengths

    # pad to max length in the batch
    padded_audios = pad_sequence(audios, batch_first=True)  # (batch, max_len, channels)

    # put back to (batch, channels, time)
    padded_audios = padded_audios.permute(0, 2, 1)

    return {
        "id": [item["id"] for item in batch],
        "chunk_id": [item["chunk_id"] for item in batch],
        "audio": padded_audios,
        "lengths": torch.tensor(lengths),  # store actual lengths before padding
        "sample_rate": batch[0]["sample_rate"],  # assume consistent sr
        "hit_objects": [item["hit_objects"] for item in batch],
    }


def createDataLoader(input_folder, batch_size):
    dataset = BeatmapChunkDataset(input_folder)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    return dataloader
