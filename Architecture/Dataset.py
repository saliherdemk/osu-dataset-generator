import os
import random
from collections import defaultdict

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
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
        chunk_id = random.choice(self.chunks[beatmapset])
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
        X_audio = log_mel_spectrogram.T

        chunk_start = 5000 * chunk_id
        chunk_end = chunk_start + 10000
        df = self.input_df
        df = df[
            (df["id"] == beatmap)
            & (df["time"] >= chunk_start)
            & (df["time"] <= chunk_end)
        ]
        df = df.copy()

        df["start"] = df["time"]
        df["end"] = df["time"] + df["duration"].replace(0, 1)

        frame_hop = 20
        frame_starts = np.arange(chunk_start, chunk_end + frame_hop, frame_hop)
        frame_ends = frame_starts + frame_hop

        frames = pd.DataFrame({"frame_start": frame_starts, "frame_end": frame_ends})

        intervals = df[["start", "end"]].values
        has_hit = []
        start_offsets = []
        end_offsets = []

        for fs, fe in zip(frames["frame_start"], frames["frame_end"]):
            active = [(s, e) for s, e in intervals if (fs < e) and (fe > s)]
            has_hit.append(1 if active else 0)

            start_offset = 0
            for s, e in active:
                if fs <= s < fe:
                    start_offset = s - fs
            start_offsets.append(start_offset)

            end_offset = 0
            for s, e in active:
                if fs < e <= fe:
                    end_offset = e - fs
            end_offsets.append(end_offset)

        frame_df = pd.DataFrame(
            {
                "frame_start": frame_starts,
                "frame_end": frame_ends,
                "has_hit": has_hit,
                "start_offset": start_offsets,
                "end_offset": end_offsets,
            }
        )

        has_hit = np.array(has_hit)
        start_offsets = np.array(start_offsets)
        end_offsets = np.array(end_offsets)

        frame_df.to_csv("/home/saliherdemk/try_dataset/frames.csv", index=False)

        has_hit = np.array(has_hit)
        start_offsets = np.array(start_offsets)
        end_offsets = np.array(end_offsets)

        return {
            "beatmap": beatmap,
            "chunk_id": chunk_id,
            "audio": X_audio.shape,
            "has_hit": has_hit.shape,
            "start_offsets": start_offsets.shape,
            "end_offsets": end_offsets.shape,
        }


def createDataLoader(input_folder, batch_size):
    dataset = BeatmapChunkDataset(input_folder)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
