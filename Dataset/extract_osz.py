import os
import zipfile
import argparse


def extract_osz(osz_file, output_folder):
    with zipfile.ZipFile(osz_file, "r") as zip_ref:
        zip_ref.extractall(output_folder)
    print(f"Extracted {osz_file} to {output_folder}")


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".osz"):
            osz_file_path = os.path.join(input_folder, filename)
            folder_name = os.path.splitext(filename)[0]
            folder_output_path = os.path.join(output_folder, folder_name)

            if not os.path.exists(folder_output_path):
                os.makedirs(folder_output_path)

            extract_osz(osz_file_path, folder_output_path)


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


if __name__ == "__main__":
    main()
