#!/bin/bash
set -euo pipefail

for id in $(seq 0 4 32); do
    echo "=============================="
    echo " Running pipeline for ID ${id}..."
    echo "=============================="

    python Dataset/merge_formatted_files.py \
        --folder_one=/run/media/saliherdemk/HIKVISION/merged/${id}/ \
        --folder_two=/run/media/saliherdemk/HIKVISION/merged/${id+2}/ \
        --output_folder=/home/saliherdemk/merged2/${id}
    
    # python Tokenizer/encode.py \
    #     --input_file=/home/saliherdemk/osu-dataset/${id}/formatted/formatted.csv \
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

