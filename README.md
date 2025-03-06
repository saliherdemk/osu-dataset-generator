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

# Cleaning & Reformatting

Although the `.osu` file format backward compatibilty, since it's changing constatly some of the attributes might be missing in dataset. Also quality of maps were highly increased (imo) over time so I remove the beatmaps that is created earlier than 2010. Also I analized and filled missing cells. You can observe `data_celaning.ipynb` to see the process. Note that this is totally up to you and may depends on your dataset. That's way it's not part of the data pipeline.


Now the hard part. Matching hit objects with timing points. 

example `time_points_df`

| ID         | time   | beat_length | meter        | sample_set | volume | uninherited | effects | beatmap_id |
|------------|--------|-------------|--------------|------------|--------|-------------|---------|------------|
| 2276798-0  | -138.0 | 349.854227  | 4            | 2          | 40     | 1.0         | 0.0     | 2276798    |
| 2276798-0  | 9657.0 | -133.333333 | 4            | 2          | 65     | 0.0         | 0.0     | 2276798    |
| 2276798-0  | 12106.0| -100.000000 | 4            | 2          | 65     | 0.0         | 0.0     | 2276798    |
| 2276798-0  | 12981.0| -133.333333 | 4            | 2          | 65     | 0.0         | 0.0     | 2276798    |
| 2276798-0  | 13331.0| -200.000000 | 4            | 2          | 65     | 0.0         | 0.0     | 2276798    |

example `hit_objects.df`
| ID         | Time   | Type   | X   | Y   | HitSound | Extra                                | beatmap_id |
|------------|--------|--------|-----|-----|----------|--------------------------------------|------------|
| 2276798-0  | 1786   | spinner| 256 | 192 | 0        | 9308                                 | 2276798    |
| 2276798-0  | 9657   | slider | 368 | 384 | 0        | B\|322:342\|307:219\|349:162\|445:131\|620:183\|543:... | 2276798    |
| 2276798-0  | 11057  | slider | 253 | 165 | 0        | P\|269:98\|301:13                       | 2276798    |
| 2276798-0  | 11582  | circle | 192 | 0   | 8        | 0                                    | 2276798    |
| 2276798-0  | 11931  | slider | 192 | 0   | 0        | P\|200:17\|207:35                       | 2276798    |


You can read what are those represents from [osu wiki!](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29)

We want to get current time attributes for each hit objects. So we need to get the latest timing point before that particular hit object and extract attributes from there. For uninherited timing points beat_length represents ms_per_beat. For inherited ones, it represents the slider multiplier. So more than one timing points may effect the same hit object. That's why we need to create columns for each possible timing points. 


