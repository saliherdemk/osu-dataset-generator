import argparse
import os

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def get_chunks_for_hit(hit_start_ms, duration_ms, chunk_length_ms, chunk_stride_ms):
    hit_end_ms = hit_start_ms + duration_ms
    chunk_starts = np.arange(0, hit_end_ms + chunk_stride_ms, chunk_stride_ms)
    chunk_ids = []
    for i, chunk_start in enumerate(chunk_starts):
        chunk_end = chunk_start + chunk_length_ms
        if hit_end_ms > chunk_start and hit_start_ms < chunk_end:
            chunk_ids.append(i)
    return chunk_ids


def normalize(df_exploded):
    chunk_length_ms = 10000

    df_exploded["hit_start_rel"] = df_exploded["hit_start_rel"] / chunk_length_ms
    df_exploded["hit_end_rel"] = df_exploded["hit_end_rel"] / chunk_length_ms

    type_onehot = pd.get_dummies(df_exploded["type"], prefix="type").astype(int)
    df_exploded = df_exploded.drop(columns=["type"])
    df_exploded = pd.concat([df_exploded, type_onehot], axis=1)

    df_exploded["difficulty_rating"] = df_exploded["difficulty_rating"] / 10

    return df_exploded


def divide_tokens(df, hit_obj_count=5):
    df["end"] = df["time"] + df["duration"]

    df["group"] = df.groupby("id").cumcount() // hit_obj_count

    return df


def divide_audio(audio_folder, embedding_folder, chunk_length_sec, overlap_sec):
    target_sr = 24000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(embedding_folder, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        "m-a-p/MERT-v1-95M", trust_remote_code=True
    )
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    model.eval()
    model.to(device)

    audio_files = [f for f in os.listdir(audio_folder)]

    for audio_file in tqdm(audio_files, desc="Processing audio"):
        audio_path = os.path.join(audio_folder, audio_file)
        beatmap_id = os.path.splitext(audio_file)[0]

        y, sr = librosa.load(audio_path, sr=target_sr)
        total_sec = y.shape[0] / sr
        chunk_stride = chunk_length_sec - overlap_sec
        chunk_starts = np.arange(0, total_sec, chunk_stride)

        for chunk_idx, start_sec in enumerate(chunk_starts):
            end_sec = start_sec + chunk_length_sec
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            chunk_audio = y[start_sample:end_sample].astype(np.float32)
            if len(chunk_audio) == 0:
                continue

            inputs = processor(
                raw_speech=chunk_audio, sampling_rate=sr, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            all_layer_hidden_states = torch.stack(outputs.hidden_states)
            layer_reduced = all_layer_hidden_states.mean(0)

            save_name = f"{beatmap_id}_chunk{chunk_idx}.pt"
            save_path = os.path.join(embedding_folder, save_name)
            torch.save(layer_reduced, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)

    args = parser.parse_args()

    formatted_df = pd.read_csv(os.path.join(args.input_folder, "formatted.csv"))
    chunked_df = divide_tokens(formatted_df)

    # audio_folder = os.path.join(args.input_folder, "audio")
    # embedding_folder = os.path.join(args.output_folder, "audio")
    # os.makedirs(embedding_folder, exist_ok=True)
    #
    # divide_audio(audio_folder, embedding_folder, chunk_length_sec, overlap_sec)

    chunked_df.to_csv(os.path.join(args.output_folder, "chunked.csv"), index=False)


if __name__ == "__main__":
    main()
