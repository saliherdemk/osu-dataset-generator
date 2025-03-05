# Dataset Pipeline

Download your current beatmapset using [this](https://github.com/saliherdemk/osu-lazer-backup) tool. Now you have .osz files for your beatmapsets.

Use `extract_osz.py` to unzip them.

Some beatmaps might not be extracted correctly. In that case you will see a warning message for that beatmapset. Extract manually and move files to the corresponding path.

```
python Dataset/extract_osz.py --input_folder=/home/saliherdemk/songs --output_folder=/home/saliherdemk/extracted
```

Now you have .osu file and corresponding audio file for every beatmap that you have. Use `generate_dataset.py` script to generate your dataset.


```
python Dataset/generate_dataset.py --input_folder=/home/saliherdemk/extracted --dataset_path=/home/saliherdemk/dataset
```

That will generate 3 files and one folder.

`beatmaps.csv`, `hit_objects.csv`, `timing_points.csv` and `audio` folder which contains only the song audio file.

Filter ranked maps. To do that, you need an OAuth key from osu. Get your `client id` and your `client secret` and paste into the `.env` file which you will create in base folder. 

```
python Dataset/filter_ranked.py --dataset_folder=/home/saliherdemk/dataset
```

Some of the audio files might be corrupted or not ready to process. Fix those.

```
python Dataset/fix_corrupted_audio.py --dataset_folder=/home/saliherdemk/dataset
```

# Merge datasets
If you collect more data later, you can merge them. After you processed your second dataset, merge them with `merge_dataset.py` script.

```
python Dataset/merge_datasets.py --dataset_one=/home/saliherdemk/dataset --dataset_two=/home/saliherdemk/dataset2
```

This will merge those datasets and overwrite to the dataset one.

