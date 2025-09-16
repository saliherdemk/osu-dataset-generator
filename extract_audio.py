import argparse
import os
import shutil

from tqdm import tqdm


def extract_audio(input_folder, output_folder):
    beatmap_folders = [
        entry
        for entry in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, entry))
    ]
    with tqdm(total=len(beatmap_folders), desc="Copying audio files") as pbar:

        for entry in beatmap_folders:
            entry_path = os.path.join(input_folder, entry)
            audio_files = [
                f
                for f in os.listdir(entry_path)
                if f.lower().endswith((".mp3", ".ogg"))
            ]
            audio_folder = os.path.join(output_folder, entry.split("-")[1])
            os.makedirs(audio_folder, exist_ok=True)

            try:
                osu_file = [
                    f for f in os.listdir(entry_path) if f.lower().endswith((".osu"))
                ][0]
            except:
                print(
                    "Couldn't find .osu file in",
                    entry_path,
                    "You may need to add manually.",
                )
                continue

            audio_files = {f.lower(): f for f in os.listdir(entry_path)}

            audio_filename = ""

            with open(os.path.join(entry_path, osu_file), "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("AudioFilename"):
                    audio_filename = line.split(":")[1].strip().lower()
                    continue

            audio_file = audio_files[audio_filename.lower()]

            shutil.move(
                os.path.join(entry_path, audio_file),
                os.path.join(audio_folder, audio_file),
            )

            pbar.update(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    args = parser.parse_args()

    extract_audio(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
