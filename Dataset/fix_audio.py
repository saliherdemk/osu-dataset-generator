import os
import subprocess
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torchaudio
from tqdm import tqdm

audio_folder = "/home/saliherdemk/Downloads/archive/audio/audio/"

for f in tqdm(os.listdir(audio_folder)):
    path = os.path.join(audio_folder, f)

    try:
        _ = torchaudio.info(path)

    except Exception as e:
        print(f"[Fixing corrupted metadata] {path}")
        base, ext = os.path.splitext(path)
        temp_path = base + "_tmp" + ext

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                path,
                "-map_metadata",
                "-1",
                "-c",
                "copy",
                temp_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        print(result)

        os.replace(temp_path, path)

print("✅ Audio metadata cleanup completed.")
