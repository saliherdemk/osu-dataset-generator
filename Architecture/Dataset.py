import os
import random
from collections import defaultdict

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset


class BeatmapChunkDataset(Dataset):
    def __init__(self, input_folder, chunk_length_sec=10, step_length_sec=5, sr=22050):
        self.input_df = pd.read_csv(os.path.join(input_folder, "formatted.csv"))
        self.beatmaps = list(self.input_df["id"].unique())
        self.audio_folder = os.path.join(input_folder, "audio")
        self.chunks_folder = os.path.join(input_folder, "chunks")
        os.makedirs(self.chunks_folder, exist_ok=True)

        self.chunk_size = int(chunk_length_sec * sr)
        self.step_size = int(step_length_sec * sr)
        self.sr = sr
        self.chunks = defaultdict(list)

        self.divide_audio()

    def save_audio_chunk(self, base_name, chunk_idx, chunk_audio):
        if len(chunk_audio) < self.chunk_size:
            pad_len = self.chunk_size - len(chunk_audio)
            chunk_audio = np.pad(chunk_audio, (0, pad_len), mode="constant")

        chunk_filename = f"{base_name}_{chunk_idx}.wav"
        chunk_path = os.path.join(self.chunks_folder, chunk_filename)

        sf.write(chunk_path, chunk_audio, self.sr)
        self.chunks[base_name].append(chunk_idx)

    def divide_audio(self):
        files = os.listdir(self.audio_folder)
        for f in files:
            full_path = os.path.join(self.audio_folder, f)
            base_name, _ = os.path.splitext(f)

            y, _ = librosa.load(full_path, sr=self.sr)
            total_samples = len(y)

            start = 0
            chunk_idx = 0
            while start + self.chunk_size <= total_samples:
                end = start + self.chunk_size
                chunk_audio = y[start:end]

                self.save_audio_chunk(base_name, chunk_idx, chunk_audio)

                start += self.step_size
                chunk_idx += 1

            if start < total_samples:
                last_chunk = y[start:]
                self.save_audio_chunk(base_name, chunk_idx, last_chunk)

    def __len__(self):
        return len(self.beatmaps)

    def __getitem__(self, idx):
        beatmap = self.beatmaps[idx]
        beatmapset = beatmap.split("-")[0]
        beatmap_data = [
            self.get_chunk_data(beatmap, chunk_id)
            for chunk_id in self.chunks[beatmapset]
        ]
        return beatmap_data

    def get_chunk_data(self, beatmap, chunk_id):
        beatmapset = beatmap.split("-")[0]
        audio_file = os.path.join(self.chunks_folder, f"{beatmapset}_{chunk_id}.wav")

        y, _ = librosa.load(audio_file, sr=self.sr)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=512,
            hop_length=441,  # 20 ms
            n_mels=64,
        )

        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        audio = log_mel_spectrogram.T

        chunk_start = 5000 * chunk_id
        chunk_end = chunk_start + 10000
        df = self.input_df
        difficulty_rating = np.full(
            audio.shape, df["difficulty_rating"].iloc[0], dtype=float
        )
        df = df[
            (df["id"] == beatmap)
            & (df["time"] >= chunk_start)
            & (df["time"] <= chunk_end)
        ]
        df = df.copy()

        frame_hop = 20
        frame_starts = np.arange(chunk_start, chunk_end + frame_hop, frame_hop)
        frame_ends = frame_starts + frame_hop

        fs = frame_starts[:, np.newaxis]
        fe = frame_ends[:, np.newaxis]

        if df.empty:
            num_frames = len(frame_starts)
            has_hit = np.zeros(num_frames, dtype=int)
            start_offsets = np.zeros(num_frames, dtype=float)
            end_offsets = np.zeros(num_frames, dtype=float)
        else:
            starts = df["time"].values
            ends = (df["time"] + df["duration"].replace(0, 1)).values

            overlaps = (fs < ends) & (fe > starts)

            has_hit = (overlaps.sum(axis=1) > 0).astype(int)

            start_conditions = (fs <= starts) & (starts < fe) & overlaps
            marked_offsets_per_frame = np.where(
                start_conditions, starts - fs.flatten()[:, np.newaxis], 0
            ).max(axis=1)
            start_offsets = np.where(
                start_conditions.any(axis=1), marked_offsets_per_frame, 0
            )

            end_conditions = (fs < ends) & (ends <= fe) & overlaps
            end_offsets = np.where(
                end_conditions.any(axis=1),
                np.where(end_conditions, ends - fs.flatten()[:, np.newaxis], 0).max(
                    axis=1
                ),
                0,
            )

        frame_df = pd.DataFrame(
            {
                "frame_start": frame_starts,
                "frame_end": frame_ends,
                "has_hit": has_hit,
                "start_offset": start_offsets,
                "end_offset": end_offsets,
            }
        )

        frame_df.to_csv("/home/saliherdemk/try_dataset/frames.csv", index=False)

        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        diff_tensor = torch.tensor(difficulty_rating, dtype=torch.float32)

        has_hit = torch.tensor(has_hit, dtype=torch.int32)
        start_offsets = torch.tensor(start_offsets, dtype=torch.float32)
        end_offsets = torch.tensor(end_offsets, dtype=torch.float32)

        return {
            "beatmap": beatmap,
            "chunk_id": chunk_id,
            "audio": audio_tensor,
            "has_hit": has_hit,
            "start_offsets": start_offsets,
            "end_offsets": end_offsets,
            "difficulty_rating": diff_tensor,
        }


def collate_fn(batch):
    max_chunks = max(len(item) for item in batch)

    batch_audio = []
    batch_has_hit = []
    batch_start_offsets = []
    batch_end_offsets = []
    batch_diff = []

    for item in batch:
        n_chunks = len(item)

        audio_shape = item[0]["audio"].shape
        has_hit_shape = item[0]["has_hit"].shape
        start_shape = item[0]["start_offsets"].shape
        end_shape = item[0]["end_offsets"].shape
        diff_shape = item[0]["difficulty_rating"].shape

        pad_audio = torch.zeros(
            (max_chunks - n_chunks, *audio_shape), dtype=torch.float32
        )
        pad_has_hit = torch.zeros(
            (max_chunks - n_chunks, *has_hit_shape), dtype=torch.long
        )
        pad_start = torch.zeros(
            (max_chunks - n_chunks, *start_shape), dtype=torch.float32
        )
        pad_end = torch.zeros((max_chunks - n_chunks, *end_shape), dtype=torch.float32)
        pad_diff = torch.zeros(
            (max_chunks - n_chunks, *diff_shape), dtype=torch.float32
        )

        audios = torch.stack([chunk["audio"] for chunk in item], dim=0)
        has_hits = torch.stack([chunk["has_hit"] for chunk in item], dim=0)
        start_offsets = torch.stack([chunk["start_offsets"] for chunk in item], dim=0)
        end_offsets = torch.stack([chunk["end_offsets"] for chunk in item], dim=0)
        diffs = torch.stack([chunk["difficulty_rating"] for chunk in item], dim=0)

        audios = torch.cat([audios, pad_audio], dim=0)
        has_hits = torch.cat([has_hits, pad_has_hit], dim=0)
        start_offsets = torch.cat([start_offsets, pad_start], dim=0)
        end_offsets = torch.cat([end_offsets, pad_end], dim=0)
        diffs = torch.cat([diffs, pad_diff], dim=0)

        batch_audio.append(audios)
        batch_has_hit.append(has_hits)
        batch_start_offsets.append(start_offsets)
        batch_end_offsets.append(end_offsets)
        batch_diff.append(diffs)

    batch_audio = torch.stack(batch_audio, dim=0)
    batch_has_hit = torch.stack(batch_has_hit, dim=0)
    batch_start_offsets = torch.stack(batch_start_offsets, dim=0)
    batch_end_offsets = torch.stack(batch_end_offsets, dim=0)
    batch_diff = torch.stack(batch_diff, dim=0)

    return {
        "audio": batch_audio,
        "difficulty_rating": batch_diff,
        "has_hit": batch_has_hit,
        "start_offsets": batch_start_offsets,
        "end_offsets": batch_end_offsets,
    }


def createDataLoader(input_folder, batch_size):
    dataset = BeatmapChunkDataset(input_folder)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    return dataloader
