import argparse
import os

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from config import CHUNK_LENGTH_SEC, SR, STEP_LENGTH_SEC

chunk_size = int(CHUNK_LENGTH_SEC * SR)
step_size = int(STEP_LENGTH_SEC * SR)


def save_audio_chunk(base_name, chunk_idx, chunk_audio, output_folder):
    if len(chunk_audio) < chunk_size:
        pad_len = chunk_size - len(chunk_audio)
        chunk_audio = np.pad(chunk_audio, (0, pad_len), mode="constant")

    chunk_filename = f"{base_name}_{chunk_idx}.wav"
    chunk_path = os.path.join(output_folder, chunk_filename)

    sf.write(chunk_path, chunk_audio, SR)


def divide_audio(input_folder, output_folder):
    os.makedirs(output_folder)
    files = os.listdir(input_folder)
    for f in tqdm(files):
        full_path = os.path.join(input_folder, f)
        base_name, _ = os.path.splitext(f)

        y, _ = librosa.load(full_path, sr=SR)
        total_samples = len(y)

        start = 0
        chunk_idx = 0
        while start + chunk_size <= total_samples:
            end = start + chunk_size
            chunk_audio = y[start:end]

            save_audio_chunk(base_name, chunk_idx, chunk_audio, output_folder)

            start += step_size
            chunk_idx += 1

        if start < total_samples:
            last_chunk = y[start:]
            save_audio_chunk(base_name, chunk_idx, last_chunk, output_folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)

    args = parser.parse_args()

    divide_audio(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
