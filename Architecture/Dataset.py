import os

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from config import CHUNK_LENGTH_SEC, HOP_LENGTH, N_FFT, N_MELS, SR, STEP_LENGTH_SEC


class PrecomputedDataset(Dataset):
    def __init__(self, input_folder):
        self.files = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.endswith(".npz")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        spectrogram = torch.from_numpy(data["spectrogram"].astype(np.float32))
        labels = torch.from_numpy(data["labels"].astype(np.float32))
        difficulty = torch.from_numpy(data["difficulty"].astype(np.float32))
        return spectrogram, difficulty, labels


def createDataLoader(input_folder, batch_size):
    dataset = PrecomputedDataset(input_folder)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
    )

    return dataloader


# Look for scipy binary dilation
def prepare_audio_for_prediction(audio_path):
    wf_np, _ = librosa.load(audio_path, sr=SR, mono=True)
    wf = torch.from_numpy(wf_np).unsqueeze(0)
    total_duration = wf_np.shape[0] / SR

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
