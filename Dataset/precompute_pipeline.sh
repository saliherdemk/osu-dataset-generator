
#!/bin/bash
set -euo pipefail


for id in $(seq 14 34); do
    echo "=============================="
    echo " Running pipeline for ID ${id}..."
    echo "=============================="

    dataset_path="/run/media/saliherdemk/HIKVISION/osu-dataset/${id}"


    python Dataset/fix_audio.py --dataset_path=${dataset_path}

    python Dataset/precompute_mel.py --dataset_path=${dataset_path} 
    
done

echo "✅ All pipelines completed successfully!"

