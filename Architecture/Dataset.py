import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from config import CHUNK_LENGTH_SEC, HOP_LENGTH, N_FFT, N_MELS, SR, STEP_LENGTH_SEC


class BeatmapChunkDataset(Dataset):
    def __init__(self, input_folder):
        self.input_df = pd.read_csv(os.path.join(input_folder, "formatted.csv"))
        self.audio_folder = os.path.join(input_folder, "audio")
        self.chunks = self.get_chunks(input_folder)

        self.audio_cache = {}
        self.cache_order = []
        self.max_cache_size = 20

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            center=False,
            power=2.0,
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

        self.resampler = None

        self.grouped_data = {
            beatmap_id: group for beatmap_id, group in self.input_df.groupby("id")
        }
        self.diff_ratings = dict(
            zip(self.input_df["id"], self.input_df["difficulty_rating"])
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        beatmap_id, chunk_start_sec, chunk_end_sec = self.chunks[idx]
        beatmapset_id = beatmap_id.split("-")[0]
        chunk_audio = self.get_audio_chunk(
            beatmapset_id, chunk_start_sec, chunk_end_sec
        )
        hit_obj_data, diff_rating = self.get_chunk_data(
            beatmap_id, chunk_start_sec, chunk_end_sec
        )

        chunk_audio = chunk_audio.clone().detach().float().unsqueeze(0)
        diff_rating = torch.tensor(diff_rating, dtype=torch.float32).unsqueeze(0)
        hit_obj_data = torch.tensor(hit_obj_data, dtype=torch.float32)

        return chunk_audio, diff_rating, hit_obj_data

    def find_audio(self, beatmapset_id):
        mp3 = os.path.join(self.audio_folder, f"{beatmapset_id}.mp3")
        ogg = os.path.join(self.audio_folder, f"{beatmapset_id}.ogg")
        if os.path.exists(mp3):
            return mp3
        if os.path.exists(ogg):
            return ogg
        raise FileNotFoundError

    def get_chunks(self, input_folder):
        chunks_file = os.path.join(input_folder, "chunks.csv")
        if os.path.exists(chunks_file):
            print("Loading cached chunk metadata...")
            df = pd.read_csv(chunks_file)
            return [
                (row.id, float(row.start), float(row.end))
                for row in df.itertuples(index=False)
            ]

        chunks = []
        for beatmap_id in self.input_df["id"].unique():
            beatmapset_id = beatmap_id.split("-")[0]
            audio_path = self.find_audio(beatmapset_id)

            info = torchaudio.info(audio_path)
            total_samples = info.num_frames
            sr = info.sample_rate
            total_duration = total_samples / sr

            for start in np.arange(
                0, total_duration - CHUNK_LENGTH_SEC, STEP_LENGTH_SEC
            ):
                end = min(start + CHUNK_LENGTH_SEC, total_duration)
                chunks.append((beatmap_id, start, end))

        chunks_df = pd.DataFrame(chunks, columns=["id", "start", "end"])
        chunks_df.to_csv(chunks_file, index=False)
        return chunks

    def load_full_audio(self, beatmapset_id):
        if beatmapset_id in self.audio_cache:
            self.cache_order.remove(beatmapset_id)
            self.cache_order.append(beatmapset_id)
            return self.audio_cache[beatmapset_id]

        audio_path = self.find_audio(beatmapset_id)
        wf, sr = torchaudio.load(audio_path)

        if sr != SR:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(sr, SR)
            wf = self.resampler(wf)

        wf = torch.mean(wf, dim=0, keepdim=True)

        if len(self.audio_cache) >= self.max_cache_size:
            oldest = self.cache_order.pop(0)
            del self.audio_cache[oldest]

        self.audio_cache[beatmapset_id] = wf
        self.cache_order.append(beatmapset_id)

        return wf

    def get_audio_chunk(self, beatmapset_id, chunk_start, chunk_end):
        full_audio = self.load_full_audio(beatmapset_id)

        start_sample = int(chunk_start * SR)
        end_sample = int(chunk_end * SR)

        chunk_audio = full_audio[:, start_sample:end_sample]

        expected_samples = int(SR * CHUNK_LENGTH_SEC)
        if chunk_audio.shape[1] < expected_samples:
            pad_size = expected_samples - chunk_audio.shape[1]
            chunk_audio = F.pad(chunk_audio, (0, pad_size))

        # torchaudio.save("/kaggle/working/output.wav", chunk_audio, SR)

        mel_spectrogram = self.mel_transform(chunk_audio)
        log_mel_spectrogram = self.db_transform(mel_spectrogram)

        audio_features = log_mel_spectrogram.squeeze(0).T

        return audio_features

    def get_chunk_data(self, beatmap_id, chunk_start_sec, chunk_end_sec):
        chunk_start = chunk_start_sec * 1000
        chunk_end = chunk_end_sec * 1000

        df = self.grouped_data[beatmap_id]
        diff_rating = self.diff_ratings[beatmap_id]

        mask = (df["time"] >= chunk_start) & (df["time"] <= chunk_end)
        df_filtered = df[mask]

        num_frames = int(((CHUNK_LENGTH_SEC * SR) - N_FFT) / HOP_LENGTH) + 1

        if df_filtered.empty:
            return np.zeros((num_frames, 7), dtype=np.float32), diff_rating

        frame_duration = (chunk_end - chunk_start) / num_frames
        frame_starts = chunk_start + np.arange(num_frames) * frame_duration
        frame_ends = frame_starts + frame_duration

        result = np.zeros((num_frames, 9), dtype=np.float32)

        df_filtered = df_filtered.copy()
        df_filtered["end"] = df_filtered["time"] + df_filtered["duration"]

        times = df_filtered["time"].values
        ends = df_filtered["end"].values
        types = df_filtered["type"].values

        for idx in range(num_frames):
            start = frame_starts[idx]
            end = frame_ends[idx]

            start_mask = (times >= start) & (times < end)
            end_mask = (ends >= start) & (ends < end)
            overlap_mask = (times < end) & (ends > start)

            start_indices = np.where(start_mask)[0]
            if len(start_indices) > 0:
                hit_idx = start_indices[0]
                hit_type = types[hit_idx]

                if hit_type == "circle":
                    result[idx, 0] = 1
                elif hit_type == "slider":
                    result[idx, 1] = 1
                elif hit_type == "spinner":
                    result[idx, 4] = 1
                result[idx, 7] = times[hit_idx] - start

            end_indices = np.where(end_mask)[0]
            if len(end_indices) > 0:
                hit_idx = end_indices[0]
                hit_type = types[hit_idx]

                if hit_type == "slider":
                    result[idx, 3] = 1
                    result[idx, 8] = ends[hit_idx] - start
                elif hit_type == "spinner":
                    result[idx, 6] = 1
                    result[idx, 8] = ends[hit_idx] - start

            overlap_indices = np.where(overlap_mask)[0]
            for hit_idx in overlap_indices:
                if types[hit_idx] == "slider":
                    result[idx, 2] = 1
                elif types[hit_idx] == "spinner":
                    result[idx, 5] = 1

        # df = pd.DataFrame(result)
        # df["start"] = frame_starts
        # df["end"] = frame_ends

        # df.to_csv("/kaggle/working/a.csv", index=False)
        # print(beatmap_id, diff_rating, result)

        return result[:, :7], diff_rating


def createDataLoader(input_folder, batch_size):
    dataset = BeatmapChunkDataset(input_folder)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    return dataloader
