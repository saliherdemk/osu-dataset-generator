
#!/bin/bash


python /home/saliherdemk/projects/osu-dataset-generator/Dataset/format_dataset.py --dataset_path=/mnt/L-HDD/try_dataset

python /home/saliherdemk/projects/osu-dataset-generator/Dataset/split_to_columns.py --input_file=/mnt/L-HDD/try_dataset/formatted.csv --output_file=/mnt/L-HDD/try_dataset/splitted.csv

python /home/saliherdemk/projects/osu-dataset-generator/Dataset/get_mfcc_parameters.py --file=/mnt/L-HDD/try_dataset/splitted.csv --output_file=//mnt/L-HDD/try_dataset/mfcc.json

python /home/saliherdemk/projects/osu-dataset-generator/Dataset/normalize.py --input_file=/mnt/L-HDD/try_dataset/splitted.csv --output_file=/mnt/L-HDD/try_dataset/normalized.csv --mfcc_parameters=/mnt/L-HDD/try_dataset/mfcc.json

python /home/saliherdemk/projects/osu-dataset-generator/Dataset/denormalize.py --input_file=/mnt/L-HDD/try_dataset/normalized.csv --output_file=/mnt/L-HDD/try_dataset/denormalized.csv --audio_path=/mnt/L-HDD/try_dataset/audio/2338303/audio.mp3
