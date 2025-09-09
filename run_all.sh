#!/bin/bash
set -euo pipefail

for id in $(seq 2 2 23); do
    echo "=============================="
    echo " Running pipeline for ID ${id}..."
    echo "=============================="

    python Dataset/merge_dataset.py \
        --folder_one=/home/saliherdemk/osu-dataset/${id}/formatted/ \
        --folder_two=/home/saliherdemk/osu-dataset/${id+1}/formatted \
        --output_folder=/home/saliherdemk/merged${id}
    
    # python Tokenizer/encode.py \
    #     --input_file=/home/salih+erdemk/osu-dataset/${id}/formatted/formatted.csv \
    #     --output_file=/home/saliherdemk/osu-dataset/${id}/formatted/encoded.csv \
    #     --mel_folder=/home/saliherdemk/osu-dataset/${id}/formatted/mels



    # ./Dataset/run_pipeline.sh \
    #     "/home/saliherdemk/osu-chunks/${id}" \
    #     "/home/saliherdemk/osu-dataset/${id}" \
    #     "/home/saliherdemk/a"
    #
    # python Dataset/format_dataset.py \
    #     --dataset_path="/home/saliherdemk/osu-dataset/${id}"
    #
    # if [ -d "/home/saliherdemk/osu-dataset/${id}/audio" ]; then
    #     rm -rf "/home/saliherdemk/osu-dataset/${id}/audio"
    #     echo "Removed audio folder for ID ${id}"
    # fi
done

echo "✅ All pipelines completed successfully!"

