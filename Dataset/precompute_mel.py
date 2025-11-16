import argparse
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

import torchaudio

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from config import CHUNK_LENGTH_SEC, HOP_LENGTH, N_FFT, N_MELS, SR, STEP_LENGTH_SEC


class ComputeMelClass:
    def __init__(self, input_folder):
        dtypes = {
            "id": "string",
            "time": "float64",
            "type": "string",
            "difficulty_rating": "float16",
            "duration": "int64",
        }

        self.input_df = pd.read_csv(
            os.path.join(input_folder, "formatted.csv"),
            dtype=dtypes,
            usecols=dtypes.keys(),
        )
        self.audio_folder = os.path.join(input_folder, "audio")
        self.chunks = self.get_chunks(input_folder)

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

    def find_audio(self, beatmapset_id):
        # I hate everyone who has extention that is not .mp3 or .ogg.
        # Special hate for https://osu.ppy.sh/beatmapsets/1855918#osu/3814170 who named audio extension as .727.
        extensions = ["", ".Mp3", ".ogg", ".MP3", ".OGG", ".mp3", ".727"]
        paths = [
            os.path.join(self.audio_folder, str(beatmapset_id) + e) for e in extensions
        ]

        for p in paths:
            if os.path.exists(p):
                return p

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
        audio_path = self.find_audio(beatmapset_id)
        wf, sr = torchaudio.load(audio_path)

        if sr != SR:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(sr, SR)
            wf = self.resampler(wf)

        wf = torch.mean(wf, dim=0, keepdim=True)

        return wf

    def get_audio_chunk(self, full_audio, chunk_start, chunk_end):
        start_sample = int(chunk_start * SR)
        end_sample = int(chunk_end * SR)

        chunk_audio = full_audio[:, start_sample:end_sample]

        expected_samples = int(SR * CHUNK_LENGTH_SEC)
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

        df = self.grouped_data[beatmap_id]
        diff_rating = self.diff_ratings[beatmap_id]

        mask = (df["time"] >= chunk_start) & (df["time"] <= chunk_end)
        df_filtered = df[mask]

        num_frames = int(((CHUNK_LENGTH_SEC * SR) - N_FFT) / HOP_LENGTH) + 1

        if df_filtered.empty:
            return np.zeros((num_frames, 7), dtype=np.float32), diff_rating

        frame_duration = (CHUNK_LENGTH_SEC * 1000) / num_frames
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

        return result[:, :7], diff_rating

    def save_mel(self, output_folder):

        chunks_by_set = defaultdict(list)
        for b_data in self.chunks:
            beatmap_id, chunk_start_sec, chunk_end_sec = b_data
            beatmapset_id = beatmap_id.split("-")[0]
            chunks_by_set[beatmapset_id].append(b_data)

        for beatmapset_id, chunk_list in tqdm(chunks_by_set.items()):
            full_audio = self.load_full_audio(beatmapset_id)

            for b_data in chunk_list:
                beatmap_id, chunk_start_sec, chunk_end_sec = b_data

                chunk_audio = self.get_audio_chunk(
                    full_audio, chunk_start_sec, chunk_end_sec
                )
                hit_obj_data, diff_rating = self.get_chunk_data(
                    beatmap_id, chunk_start_sec, chunk_end_sec
                )

                chunk_path = os.path.join(
                    output_folder, f"{beatmap_id}_{chunk_start_sec}_{chunk_end_sec}.npz"
                )

                np.savez_compressed(
                    chunk_path,
                    spectrogram=chunk_audio.numpy().astype(np.float16),
                    labels=hit_obj_data.astype(np.float16),
                    difficulty=np.array([diff_rating], dtype=np.float16),
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", required=True, help="Path to dataset root folder."
    )
    args = parser.parse_args()

    mel_class = ComputeMelClass(args.dataset_path)
    precomputed_folder = os.path.join(args.dataset_path, "precomputed")
    os.makedirs(precomputed_folder, exist_ok=True)
    mel_class.save_mel(precomputed_folder)


if __name__ == "__main__":
    main()
