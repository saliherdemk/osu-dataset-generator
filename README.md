# Dataset Pipeline

Download your current beatmapset using [this](https://github.com/saliherdemk/osu-lazer-backup) tool. Now you have .osz files for your beatmapsets.

Use `extract_osz.py` to unzip them.


```
python Dataset/extract_osz.py --input_folder=/home/saliherdemk/songs --output_folder=/home/saliherdemk/extracted
```

Now you have .osu file and corresponding audio file for every beatmap that you own. Use `generate_dataset.py` script to generate your dataset.


```
python Dataset/generate_dataset.py --input_folder=/home/saliherdemk/extracted --dataset_path=/home/saliherdemk/dataset
```

That will generate 3 files and one folder.

`beatmaps.csv`, `hit_objects.csv`, `timing_points.csv` and `audio` folder which contains only the song audio file.

Next, retrieve beatmap metadata and add it to the dataset. For this, you need an OAuth key from osu. Get your client id and client secret, then paste them into a `.env` file, which you will create in the base folder.

 
```
python Dataset/add_beatmaps_metadata.py --dataset_folder=/home/saliherdemk/dataset
```

Filter ranked maps and remove old maps. (ie > 2010) 
```
python Dataset/filter_ranked.py --dataset_folder=/home/saliherdemk/dataset
```

Some of the audio files might be corrupted or not ready for processing. Fix those.

```
python Dataset/fix_corrupted_audio.py --dataset_folder=/home/saliherdemk/dataset
```

# Run Pipeline
You can run everything at once using the `run_pipeline.sh` script.

```
chmod +x /Dataset/run_pipeline.sh
```

```
./Dataset/run_pipeline.sh /mnt/L-HDD/songs /mnt/L-HDD/dataset /mnt/L-HDD/temp/
```


# Reformatting

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


You can read what these attributes represent on the [osu wiki!](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29)

We need to get the current timing attributes for each hit object. To do this, we must find the latest timing point before the hit object and extract attributes from there. For uninherited timing points, `beat_length` represents `ms_per_beat`, whereas for inherited ones, it represents the `slider multiplier`. More than one timing point may affect the same hit object, so we need to create columns for each possible timing point.

For each row, we need `beat_length`, `meter`, `slider_velocity`, `sample_set`, `volume`, `effects`
values. Also we need the corresponding `MFCC` and `RMS` values for that time which we will extract from the audio file.

```
python Dataset/format_dataset.py --dataset_path=/mnt/L-HDD/dataset
```

Now you should have your `hit_objects_formatted.csv` file on your dataset folder.

`hit_objects_formatted.csv` should look like this:
| unique_id  | ID | Time | Type   | X | Y | HitSound | Extra | beatmap_id | MFCC | RMS | beat_length | meter | slider_velocity | sample_set | volume | effects |
|----|----|----|----|----|----|----|----|----|-----|----|----|----|----|----|----|----|
| 1          | 1509063-0 | 3580 | slider | 135 | 26  | 0        | B\|181:10\|217:45\|225:69... | 1509063    | [-95.64657,... -2.9250336, -12.876349] | 0.196154251694679   | 344.827586206897     | 4     | -142                 | 1          | 60     | 0       |
| 2          | 1509063-0 | 4270 | slider | 385 | 75  | 0        | P\|349:124\|353:174 | 1509063    | [-50.801365, ... 0.97146934, 1.8829639] | 0.187753155827522   | 344.827586206897     | 4     | -108.695652173913    | 3          | 60     | 0       |
| 3          | 1509063-0 | 4615 | circle | 365 | 204 | 0        |   | 1509063    | [-70.5737, ... 5.5829735, 7.2142572] | 0.145843848586082   | 344.827586206897     | 4     | -108.695652173913    | 3          | 60     | 0       |
| 4          | 1509063-0 | 4787 | circle | 365 | 204 | 0        |   | 1509063    | [-87.663124, ... 1.1762382, 0.45896247] | 0.148052349686623   | 344.827586206897     | 4     | -108.695652173913    | 3          | 60     | 0       |
| 5          | 1509063-0 | 5132 | circle | 414 | 371 | 0        |   | 1509063    | [-108.05986, ... 4.0379977, 6.4551687] | 0.108284793794155   | 344.827586206897     | 4     | -108.695652173913    | 3          | 60     | 0       |


Some of the beatmaps may not be processed properly. If you ensure that you try to process all of the rows, you can remove the remaining ones with the clear flag.


```
python Dataset/format_dataset.py --dataset_path=/mnt/L-HDD/dataset --clear
```
This will ensure that dataset is not contains any not-processed row.


## Breaks

Since we were working with hit objects, break periods were not included in the dataset. Process break points using `add_breaks.py`

```
python Dataset/add_breaks.py --dataset_path=/mnt/L-HDD/dataset
```

Once complete, you should have a `breaks.csv` file in your dataset folder. Merge it with the formatted hit objects file

```
python Dataset/add_breaks.py --dataset_path=/mnt/L-HDD/dataset --merge
```

This will merge `breaks.csv` with `hit_objects_formatted.csv`, overwriting the latter.

## Split MFCC and Curve Points Into Columns
Since we stored our MFCC's and curve points as string in one cell. We need to spread across to new columns.

```
python Dataset/split_to_columns.py --input_file=/mnt/L-HDD/dataset/hit_objects_formatted.csv --output_file=/mnt/L-HDD/dataset/splitted.csv
```

Notice that both arguments take file path.

## Normalize Dataset

```
python Dataset/normalize.py --input_file=/mnt/L-HDD/dataset/splitted.csv --output_file=/mnt/L-HDD/dataset/normalized.csv
```

You can find out more about normalization in `notebooks/normalize.ipynb` 

# Merge datasets

If you collect more data later, you can merge datasets. After processing your second dataset, merge it with `merge_datasets.py`

```
python Dataset/merge_datasets.py --dataset_one=/mnt/L-HDD/dataset1 --dataset_two=/mnt/L-HDD/dataset2 --output_file=/mnt/L-HDD/merged.csv
```

Notice that `--output_file` argument is a file path not a directory. This script assumes both dataset folders have been processed and contain a `hit_objects_formatted.csv` file. It will filter and add beatmaps from the second dataset that are not present in the first.


