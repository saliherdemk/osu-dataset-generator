
#!/bin/bash
set -euo pipefail

for id in $(seq 0 7); do
    echo "=============================="
    echo " Running pipeline for ID ${id}..."
    echo "=============================="

    python Dataset/pipeline/extract_osz.py --input_folder=/home/saliherdemk/osu-chunks/${id} --output_folder=/home/saliherdemk/osu-dataset/a/

    python extract_audio.py --input_folder=/home/saliherdemk/osu-dataset/a/ --output_folder=/home/saliherdemk/osu-dataset/audio

    rm -rf /home/saliherdemk/osu-dataset/a

done

echo "✅ All pipelines completed successfully!"

