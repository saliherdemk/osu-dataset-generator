import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import soundfile as sf

from config import CHUNK_LENGTH_SEC, HOP_LENGTH, N_FFT, N_MELS, SR


class BeatmapChunkDataset(Dataset):
    def __init__(self, input_folder):
        self.input_df = pd.read_csv(os.path.join(input_folder, "formatted.csv"))
        self.audio_folder = os.path.join(input_folder, "audio")
        self.beatmaps = list(self.input_df["id"].unique())

    def __len__(self):
        return len(self.beatmaps)

    def __getitem__(self, idx):
        beatmap_id = self.beatmaps[idx]
        beatmapset_id = beatmap_id.split("-")[0]
        chunk_audio, chunk_start, chunk_end = self.get_audio_chunk(beatmapset_id)

        has_hit_data = self.get_chunk_data(beatmap_id, chunk_start, chunk_end)
        return chunk_audio, has_hit_data

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

        max_start_offset_sec = int(total_duration_sec - CHUNK_LENGTH_SEC)

        start_offset_sec = np.random.randint(low=0, high=max_start_offset_sec)

        chunk_audio, _ = librosa.load(
            audio_path, sr=SR, offset=start_offset_sec, duration=CHUNK_LENGTH_SEC
        )

        chunk_start_ms = start_offset_sec * 1000
        chunk_end_ms = chunk_start_ms + CHUNK_LENGTH_SEC * 1000

        sf.write("/home/saliherdemk/try_dataset/a.wav", chunk_audio, SR)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=chunk_audio,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            center=False,
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        audio_features = log_mel_spectrogram.T

        return audio_features, chunk_start_ms, chunk_end_ms

    def get_chunk_data(self, beatmap_id, chunk_start, chunk_end):
        df = self.input_df
        df = df[
            (df["id"] == beatmap_id)
            & (df["time"] >= chunk_start)
            & (df["time"] <= chunk_end)
        ]
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

        result_df.to_csv("/home/saliherdemk/try_dataset/res_df.csv")

        result = torch.tensor(result_df[cols].values, dtype=torch.float32)

        return result


def createDataLoader(input_folder, batch_size):
    dataset = BeatmapChunkDataset(input_folder)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
