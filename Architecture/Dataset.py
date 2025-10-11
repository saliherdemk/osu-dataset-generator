import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ASTForAudioClassification, AutoFeatureExtractor

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import soundfile as sf

from config import CHUNK_LENGTH_SEC, HOP_LENGTH, SR


class BeatmapChunkDataset(Dataset):
    def __init__(self, input_folder):
        self.input_df = pd.read_csv(os.path.join(input_folder, "formatted.csv"))
        self.audio_folder = os.path.join(input_folder, "audio")
        self.beatmaps = list(self.input_df["id"].unique())
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

    def __len__(self):
        return len(self.beatmaps)

    def __getitem__(self, idx):
        beatmap_id = self.beatmaps[idx]
        beatmapset_id = beatmap_id.split("-")[0]
        chunk_audio, chunk_start, chunk_end = self.get_audio_chunk(beatmapset_id)
        inputs = self.feature_extractor(
            chunk_audio, sampling_rate=SR, return_tensors="pt"
        )

        has_hit_data = self.get_chunk_data(beatmap_id, chunk_start, chunk_end)
        print(inputs["input_values"].shape, has_hit_data.shape)
        return inputs, has_hit_data

    def get_audio_chunk(self, beatmapset_id):
        audio_path_mp3 = os.path.join(self.audio_folder, f"{beatmapset_id}.mp3")
        audio_path_ogg = os.path.join(self.audio_folder, f"{beatmapset_id}.ogg")

        audio_path = None
        if os.path.exists(audio_path_mp3):
            audio_path = audio_path_mp3
        elif os.path.exists(audio_path_ogg):
            audio_path = audio_path_ogg
        else:
            raise FileNotFoundError(f"No audio file for beatmapset_id: {beatmapset_id}")

        total_duration_sec = librosa.get_duration(path=audio_path)

        if total_duration_sec < CHUNK_LENGTH_SEC:
            y, _ = librosa.load(audio_path, sr=SR)
            chunk_audio = y
            start_offset_sec = 0.0
        else:
            max_start_offset_sec = total_duration_sec - CHUNK_LENGTH_SEC

            start_offset_sec = np.random.uniform(low=0.0, high=max_start_offset_sec)

            chunk_audio, _ = librosa.load(
                audio_path, sr=SR, offset=start_offset_sec, duration=CHUNK_LENGTH_SEC
            )

        chunk_start_ms = start_offset_sec * 1000
        chunk_end_ms = chunk_start_ms + CHUNK_LENGTH_SEC * 1000

        sf.write("/home/saliherdemk/ast_data/a.wav", chunk_audio, SR)

        return chunk_audio, chunk_start_ms, chunk_end_ms

    def get_chunk_data(self, beatmap_id, chunk_start, chunk_end):
        df = self.input_df
        df = df[
            (df["id"] == beatmap_id)
            & (df["time"] >= chunk_start)
            & (df["time"] <= chunk_end)
        ]
        df = df.copy()

        num_frames = 1024
        frame_duration = (chunk_end - chunk_start) / num_frames

        frame_starts = chunk_start + np.arange(num_frames) * frame_duration
        frame_ends = frame_starts + frame_duration

        has_hit = np.zeros(num_frames)

        fs = frame_starts[:, np.newaxis]
        fe = frame_ends[:, np.newaxis]

        if df.empty:
            num_frames = len(frame_starts)
            has_hit = np.zeros(num_frames, dtype=float)
            start_offsets = np.zeros(num_frames, dtype=float)
            end_offsets = np.zeros(num_frames, dtype=float)
        else:
            starts = df["time"].values
            ends = (df["time"] + df["duration"].replace(0, 1)).values

            overlaps = (fs < ends) & (fe > starts)

            has_hit = (overlaps.sum(axis=1) > 0).astype(float)

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

        frame_df.to_csv("/home/saliherdemk/ast_data/frames.csv", index=False)

        has_hit = torch.tensor(has_hit, dtype=torch.float32)

        return has_hit


def createDataLoader(input_folder, batch_size):
    dataset = BeatmapChunkDataset(input_folder)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
