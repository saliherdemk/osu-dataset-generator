#!/bin/bash

display_usage() {
    echo "Usage: $0 <songs_folder> <dataset_folder> <temp_extracted_folder>"
    exit 1
}

check_temp_directory() {
    if [ -d "$TEMP_EXTRACTED_FOLDER" ]; then
        echo "Error: Temporary folder '$TEMP_EXTRACTED_FOLDER' already exists."
        exit 1
    fi
}

if [ "$#" -ne 3 ]; then
    display_usage
fi

SONGS_FOLDER="$1"
DATASET_FOLDER="$2"
TEMP_EXTRACTED_FOLDER="$3"

check_temp_directory

trap "rm -rf $TEMP_EXTRACTED_FOLDER" EXIT

mkdir -p "$TEMP_EXTRACTED_FOLDER"

echo "Using temporary extracted folder: $TEMP_EXTRACTED_FOLDER"

echo "Extracting .osz files..."
python Dataset/pipeline/extract_osz.py --input_folder="$SONGS_FOLDER" --output_folder="$TEMP_EXTRACTED_FOLDER"

echo "Generating dataset..."
python Dataset/pipeline/generate_dataset.py --input_folder="$TEMP_EXTRACTED_FOLDER" --dataset_path="$DATASET_FOLDER"

echo "Adding beatmap metadata..."
python Dataset/pipeline/add_beatmaps_metadata.py --dataset_folder="$DATASET_FOLDER"

echo "Filtering ranked beatmaps..."
python Dataset/pipeline/filter_ranked.py --dataset_folder="$DATASET_FOLDER"

echo "Fixing corrupted audio..."
python Dataset/pipeline/fix_corrupted_audio.py --dataset_folder="$DATASET_FOLDER"

echo "Pipeline completed successfully!"

