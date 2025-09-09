import argparse
import os
import shutil
import zipfile

from tqdm import tqdm


def extract_osz(osz_file, output_folder):
    try:
        with zipfile.ZipFile(osz_file, "r") as zip_ref:
            zip_ref.extractall(output_folder)
    except zipfile.BadZipFile:
        shutil.rmtree(output_folder)
        print(osz_file, "Skipped. Corrupted .osz file")
        pass


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    osz_files = [
        f
        for f in os.listdir(input_folder)
        if f.endswith(".osz")
        and not os.path.exists(os.path.join(output_folder, os.path.splitext(f)[0]))
    ]

    with tqdm(total=len(osz_files), desc="Extracting", unit="file") as pbar:
        for filename in osz_files:
            osz_file_path = os.path.join(input_folder, filename)
            folder_name = os.path.splitext(filename)[0]
            folder_output_path = os.path.join(output_folder, folder_name)

            if not os.path.exists(folder_output_path):
                os.makedirs(folder_output_path)

            extract_osz(osz_file_path, folder_output_path)
            pbar.update(1)


def remove_banned_content(input_folder):
    song_folders = os.listdir(input_folder)

    for song_folder in song_folders:
        song_path = os.path.join(input_folder, song_folder)

        osu_files = [file for file in os.listdir(song_path) if file.endswith(".osu")]
        with open(os.path.join(song_path, osu_files[0]), "r", encoding="utf-8") as f:
            lines = f.readlines()

        audio_filename = None
        for line in lines:
            if line.startswith("AudioFilename"):
                audio_filename = line.split(":", 1)[1].strip()
                break

        norm_audio = audio_filename.strip().lower()
        norm_files = [f.strip().lower() for f in os.listdir(song_path)]

        if norm_audio not in norm_files:
            shutil.rmtree(os.path.join(song_path))
            print(f"Missing audio in {song_folder} → {repr(audio_filename)}. Removed")


def main():
    parser = argparse.ArgumentParser(
        description="Extract .osz files into corresponding folders."
    )
    parser.add_argument(
        "--input_folder", help="Path to the folder containing .osz files."
    )
    parser.add_argument(
        "--output_folder",
        help="Path to the folder where extracted files will be stored.",
    )

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)
    remove_banned_content(args.output_folder)


if __name__ == "__main__":
    main()
