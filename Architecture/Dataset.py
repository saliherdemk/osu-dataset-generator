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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            center=False,
            power=2.0,
        ).to(self.device)

        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        ).to(self.device)

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
        diff_rating = torch.tensor(
            diff_rating, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        hit_obj_data = torch.tensor(
            hit_obj_data, dtype=torch.float32, device=self.device
        )

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

        # chunks_df = pd.DataFrame(chunks, columns=["id", "start", "end"])
        # chunks_df.to_csv(chunks_file, index=False)
        return chunks

    def get_audio_chunk(self, beatmapset_id, chunk_start, chunk_end):
        audio_path = self.find_audio(beatmapset_id)

        wf, sr = torchaudio.load(audio_path)
        wf = wf.to(self.device)

        if sr != SR:
            resampler = torchaudio.transforms.Resample(sr, SR).to(self.device)
            wf = resampler(wf)
            sr = SR

        wf = torch.mean(wf, dim=0, keepdim=True)

        start_sample = int(chunk_start * sr)
        end_sample = int(chunk_end * sr)
        chunk_audio = wf[:, start_sample:end_sample]

        expected_samples = int(sr * CHUNK_LENGTH_SEC)
        if chunk_audio.shape[1] < expected_samples:
            pad_size = expected_samples - chunk_audio.shape[1]
            chunk_audio = F.pad(chunk_audio, (0, pad_size))

        mel_spectrogram = self.mel_transform(chunk_audio)
        log_mel_spectrogram = self.db_transform(mel_spectrogram)

        audio_features = log_mel_spectrogram.squeeze(0).T

        return audio_features

    def get_chunk_data(self, beatmap_id, chunk_start_sec, chunk_end_sec):
        chunk_start = chunk_start_sec * 1000
        chunk_end = chunk_end_sec * 1000

        df = self.input_df
        df = df[df["id"] == beatmap_id]
        diff_rating = df["difficulty_rating"].iloc[0]
        df = df[(df["time"] >= chunk_start) & (df["time"] <= chunk_end)]
        df = df.copy()

        num_frames = int(((CHUNK_LENGTH_SEC * SR) - N_FFT) / HOP_LENGTH) + 1
        frame_duration = (chunk_end - chunk_start) / num_frames

        frame_starts = chunk_start + np.arange(num_frames) * frame_duration
        frame_ends = frame_starts + frame_duration

        cols = [
            "start",
            "end",
            "is_circle",
            "is_slider_start",
            "is_slider_end",
            "is_spinner_start",
            "is_spinner_end",
            "start_offset",
            "end_offset",
        ]

        result_df = pd.DataFrame(columns=cols)

        result_df["start"] = frame_starts
        result_df["end"] = frame_ends

        for col in cols[2:]:
            result_df[col] = 0

        df["end"] = df["time"] + df["duration"]

        for i, row in result_df.iterrows():
            start, end = row["start"], row["end"]

            hit_start_in_frame = df[(df["time"] >= start) & (df["time"] < end)]
            hit_end_in_frame = df[(df["end"] >= start) & (df["end"] < end)]

            total_hits_in_frame = pd.concat(
                [hit_start_in_frame, hit_end_in_frame]
            ).drop_duplicates()
            if len(total_hits_in_frame) > 1:
                raise ValueError(
                    f"Multiple hit objects found in frame {i}: {total_hits_in_frame}"
                )

            if not hit_start_in_frame.empty:
                hit_type = hit_start_in_frame.iloc[0]["type"]

                if hit_type == "circle":
                    result_df.at[i, "is_circle"] = 1
                elif hit_type == "slider":
                    result_df.at[i, "is_slider_start"] = 1
                elif hit_type == "spinner":
                    result_df.at[i, "is_spinner_start"] = 1
                result_df.at[i, "start_offset"] = (
                    hit_start_in_frame.iloc[0]["time"] - start
                )

            if not hit_end_in_frame.empty:
                hit_type = hit_end_in_frame.iloc[0]["type"]

                if hit_type == "slider":
                    result_df.at[i, "is_slider_end"] = 1
                    result_df.at[i, "end_offset"] = (
                        hit_end_in_frame.iloc[0]["end"] - start
                    )

                elif hit_type == "spinner":
                    result_df.at[i, "is_spinner_end"] = 1
                    result_df.at[i, "end_offset"] = (
                        hit_end_in_frame.iloc[0]["end"] - start
                    )
        # result_df.to_csv("/kaggle/working/a.csv", index=False)

        return result_df[cols[2:]].values, diff_rating


def createDataLoader(input_folder, batch_size):
    dataset = BeatmapChunkDataset(input_folder)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
