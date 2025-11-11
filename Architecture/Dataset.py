import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from pandas.io.common import file_path_to_url
from torch.utils.data import DataLoader, Dataset

from config import CHUNK_LENGTH_SEC, HOP_LENGTH, N_FFT, N_MELS, SR, STEP_LENGTH_SEC


class BeatmapChunkDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.chunks = os.listdir(dataset_path)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        file_name = self.chunks[idx]
        file_path = os.path.join(self.dataset_path, file_name)

        with np.load(file_path) as data:
            chunk_audio = torch.from_numpy(data["spectrogram"]).float()
            hit_obj_data = torch.from_numpy(data["labels"]).float()
            diff_rating = torch.from_numpy(data["difficulty"]).float()

        return chunk_audio, hit_obj_data, diff_rating


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


# Look for scipy binary dilation
def prepare_audio_for_prediction(audio_path):
    wf, sr = torchaudio.load(audio_path)

    if sr != SR:
        resampler = torchaudio.transforms.Resample(sr, SR)
        wf = resampler(wf)

    wf = torch.mean(wf, dim=0, keepdim=True)

    info = torchaudio.info(audio_path)
    total_samples = info.num_frames
    sr = info.sample_rate
    total_duration = total_samples / sr

    chunks = []

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        center=False,
        power=2.0,
    )

    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    for chunk_start in np.arange(0, total_duration - CHUNK_LENGTH_SEC, STEP_LENGTH_SEC):
        chunk_end = min(chunk_start + CHUNK_LENGTH_SEC, total_duration)

        start_sample = int(chunk_start * SR)
        end_sample = int(chunk_end * SR)

        chunk_audio = wf[:, start_sample:end_sample]

        expected_samples = int(SR * CHUNK_LENGTH_SEC)
        if chunk_audio.shape[1] < expected_samples:
            pad_size = expected_samples - chunk_audio.shape[1]
            chunk_audio = F.pad(chunk_audio, (0, pad_size))

        mel_spectrogram = mel_transform(chunk_audio)
        log_mel_spectrogram = db_transform(mel_spectrogram)

        audio_features = log_mel_spectrogram.squeeze(0).T
        chunks.append(audio_features)

    return torch.stack(chunks)


def pred_to_df(probs):
    columns = [
        "start_ms",
        "end_ms",
        "is_circle",
        "is_slider_start",
        "is_slider_continue",
        "is_slider_end",
        "is_spinner_start",
        "is_spinner_continue",
        "is_spinner_end",
    ]
    rows = []

    chunk_overlap_ms = STEP_LENGTH_SEC * 1000
    num_frames = int(((CHUNK_LENGTH_SEC * SR) - N_FFT) / HOP_LENGTH) + 1
    frame_ms = (CHUNK_LENGTH_SEC * 1000) / num_frames

    num_chunks, num_frames, num_labels = probs.shape

    for i in range(num_chunks):
        chunk_start = i * chunk_overlap_ms
        for j in range(num_frames):
            start = float(chunk_start + j * frame_ms)
            end = start + frame_ms
            labels = probs[i, j].tolist()
            rows.append([start, end] + labels)

    df = pd.DataFrame(rows, columns=columns)

    return df
