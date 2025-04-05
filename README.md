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

| id          | time | beat_length        | meter | sample_set | volume | uninherited | effects | beatmap_id |
|------------|------|--------------------|-------|------------|--------|-------------|---------|------------|
| 2338303-0  | 1375 | 416.666666666667   | 4     | 2          | 65     | 1           | 0       | 2338303    |
| 2338303-0  | 1375 | -111.111111111111  | 4     | 2          | 65     | 0           | 0       | 2338303    |
| 2338303-0  | 8041 | -90.9090909090909  | 4     | 2          | 80     | 0           | 0       | 2338303    |
| 2338303-0  | 14708 | -111.111111111111  | 4     | 2          | 60     | 0           | 0       | 2338303    |
| 2338303-0  | 28041 | -100               | 4     | 2          | 70     | 0           | 0       | 2338303    |


example `hit_objects.df`
| id          | time  | type   | x   | y   | hit_sound | path                 | repeat | length        | spinner_time | new_combo | beatmap_id |
|------------|-------|--------|-----|-----|-----------|----------------------|--------|----------------|--------------|-----------|------------|
| 2338303-0  | 1375  | circle | 101 | 325 | 0         |                      | 0      | 0              | 0            | True      | 2338303    |
| 2338303-0  | 4708  | circle | 434 | 240 | 0         |                      | 0      | 0              | 0            | True      | 2338303    |
| 2338303-0  | 5541  | circle | 329 | 310 | 0         |                      | 0      | 0              | 0            | False     | 2338303    |
| 2338303-0  | 6375  | circle | 199 | 233 | 0         |                      | 0      | 0              | 0            | True      | 2338303    |
| 2338303-0  | 8041  | slider | 71  | 0   | 0         | P\|3:46\|28:141      | 1      | 90.2000027526856 | 0          | True      | 2338303    |




You can read what these attributes represent on the [osu wiki!](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29)

We need to get the current timing attributes for each hit object. To do this, we must find the latest timing point before the hit object and extract attributes from there. For uninherited timing points, `beat_length` represents `ms_per_beat`, whereas for inherited ones, it represents the `slider multiplier`. More than one timing point may affect the same hit object, so we need to create columns for each possible timing point.

For each row, we need `beat_length`, `meter`, `slider_velocity`, `sample_set`, `volume`, `effects`
values. Also we need the corresponding `MFCC` and `RMS` values for that time which we will extract from the audio file.

```
python Dataset/format_dataset.py --dataset_path=/mnt/L-HDD/dataset
```

Now you should have your `hit_objects_formatted.csv` file on your dataset folder.

`hit_objects_formatted.csv` should look like this:
| unique_id | id         | time | type   | x   | y   | hit_sound | path         | repeat | length           | spinner_time | beatmap_id | mfcc                   | rms             | beat_length        | meter | slider_velocity     | sample_set | volume | effects |
|-----------|-----------|------|--------|-----|-----|-----------|--------------|--------|------------------|--------------|------------|------------------------|----------------|--------------------|-------|---------------------|------------|--------|---------|
| 7         | 2338303-0 | 7208 | circle | 316 | 211 | 0         |              | 0      | 0                | 0            | 2338303    | [-14.43, ..., -8.69]  | 0.3607         | 416.666666666667   | 4     | -111.111111111111   | 2          | 65     | 0       |
| 8         | 2338303-0 | 7937 | circle | 101 | 13  | 0         |              | 0      | 0                | 0            | 2338303    | [-133.75, ..., -28.20] | 0.1359         | 416.666666666667   | 4     | -111.111111111111   | 2          | 65     | 0       |
| 9         | 2338303-0 | 8041 | slider | 71  | 0   | 0         | P\|3:46\|28:141 | 1      | 180.400005505371 | 0            | 2338303    | [-48.53, ..., -11.39]  | 0.2230         | 416.666666666667   | 4     | -90.9090909090909   | 2          | 80     | 0       |



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


Once complete, you should have a `dataset.csv` file in your dataset folder which looks like this:

| unique_id  | id | time | type | x | y | hit_sound  | path | repeat | length | spinner_time | new_combo | beatmap_id | mfcc | rms | beat_length | meter | slider_velocity | sample_set | volume | effects | time_interval |
|------------|-------------|-------|--------|-----|-----|------------|-------------|--------|------------|--------------|-----------|------------|----------------------------------------------------------------------|-------------|-------------|-------|-----------------|------------|--------|---------|---------------|
| 774        | 2337169-0   | 231   | slider | 86  | 114 | 0          | L\|81:47     | 1      | 67.500003  | 0            | 1         | 2337169    | [-541.5654,..., 0.] | 0.004970386 | 689.655172 | 4     | -133.33         | 2          | 40     | 0       |               |
| 775        | 2337169-0   | 920   | slider | 170 | 149 | 0          | L\|165:216   | 1      | 67.500003  | 0            | 0         | 2337169    | [-242.96797,..., -10.411037] | 0.085116714 | 689.655172 | 4     | -133.33         | 2          | 40     | 0       | 689           |
| 776        | 2337169-0   | 1610  | circle | 247 | 47  | 2          |             | 0      | 0          | 0            | 0         | 2337169    | [-196.65742,..., -14.851294] | 0.088140659 | 689.655172 | 4     | -133.33         | 2          | 40     | 0       | 690           |

Dataset should be sorted by `time` and has `time_interval` column shows the time difference between that hit object and the previous one.

## Split MFCC and Curve Points Into Columns
Since we stored our MFCC's and curve points as string in one cell. We need to spread across to new columns.

> **Warning** If you have 16GB of memory or less, you might not be able to process large datasets. Consider splitting your dataset into multiple parts (see `notebooks/split_dataset.ipynb`) and merging them after processing, or use chunks. 

```
python Dataset/split_to_columns.py --input_file=/mnt/L-HDD/dataset/dataset.csv --output_file=/mnt/L-HDD/dataset/splitted_dataset.csv
```

Notice that both arguments take file path.

## Normalize Dataset

```
python Dataset/normalize.py --input_file=/mnt/L-HDD/dataset/splitted_dataset.csv --output_file=/mnt/L-HDD/dataset/normalized.csv
```

You can find out more about normalization in `notebooks/normalize.ipynb` 
Here is the final dataset format:


# Merge datasets

If you collect more data later, you can merge datasets. After processing your second dataset, merge it with `merge_datasets.py`

```
python Dataset/merge_datasets.py --file_one=/mnt/L-HDD/dataset1.csv --file_two=/mnt/L-HDD/dataset2.csv --output_file=/mnt/L-HDD/merged.csv
```

It will filter the beatmaps based on the id column and add the beatmaps from the second dataset that are not present in the first.


## Final Data Format

`normalized.csv`
|unique_id|id       |time|x         |y                |repeat           |length          |spinner_time|beatmap_id|rms                |beat_length     |meter          |slider_velocity |volume|time_interval|mfcc_1            |mfcc_2           |mfcc_3             |mfcc_4            |mfcc_5            |mfcc_6             |mfcc_7             |mfcc_8             |mfcc_9             |mfcc_10            |mfcc_11            |mfcc_12            |mfcc_13            |mfcc_14             |mfcc_15            |mfcc_16             |mfcc_17            |mfcc_18            |mfcc_19            |mfcc_20            |path_1           |path_2           |path_3|path_4|path_5|path_6|path_7|path_8|path_9|path_10|path_11|path_12|path_13|path_14|path_15|path_16|path_17|path_18|path_19|path_20|path_21|path_22|path_23|path_24|path_25|path_26|path_27|path_28|path_29|path_30|path_31|path_32|path_33|path_34|path_35|path_36|path_37|path_38|path_39|path_40|path_41|path_42|path_43|path_44|path_45|path_46|type_break|type_circle|type_slider|type_spinner|hit_sound_0|hit_sound_2|hit_sound_4|hit_sound_6|hit_sound_8|hit_sound_10|hit_sound_12|hit_sound_14|sample_set_0.0|sample_set_1.0|sample_set_2.0|sample_set_3.0|effects_0.0|effects_1.0|curve_type_B|curve_type_E|curve_type_L|curve_type_P|new_combo_0|new_combo_1|difficulty_rating_0|difficulty_rating_2|difficulty_rating_3|difficulty_rating_4|difficulty_rating_5|difficulty_rating_6|difficulty_rating_1|difficulty_rating_7|difficulty_rating_8|difficulty_rating_9|difficulty_rating_10|difficulty_rating_11|difficulty_rating_12|
|---------|---------|----|----------|-----------------|-----------------|----------------|------------|----------|-------------------|----------------|---------------|----------------|------|-------------|------------------|-----------------|-------------------|------------------|------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|--------------------|-------------------|--------------------|-------------------|-------------------|-------------------|-------------------|-----------------|-----------------|------|------|------|------|------|------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|----------|-----------|-----------|------------|-----------|-----------|-----------|-----------|-----------|------------|------------|------------|--------------|--------------|--------------|--------------|-----------|-----------|------------|------------|------------|------------|-----------|-----------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|--------------------|--------------------|--------------------|
|774      |2337169-0|231 |0.16796875|0.296103896103896|0.693147180559945|4.22683378285826|0           |2337169   |0.00599346132646657|6.53764067231476|1.6094379124341|4.90032427327857|0.4   |0            |-1                |0                |0                  |0                 |0                 |0                  |0                  |0                  |0                  |0                  |0                  |0                  |0                  |0                   |0                  |0                   |0                  |0                  |0                  |0                  |0.145161290322581|0.114914425427873|0     |0     |0     |0     |0     |0     |0     |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |False     |False      |True       |False       |True       |False      |False      |False      |False      |False       |False       |False       |False         |False         |True          |False         |True       |False      |False       |False       |True        |False       |False      |True       |False              |False              |False              |True               |False              |False              |False              |False              |False              |False              |False               |False               |False               |
|775      |2337169-0|920 |0.33203125|0.387012987012987|0.693147180559945|4.22683378285826|0           |2337169   |0.153451637872841  |6.53764067231476|1.6094379124341|4.90032427327857|0.4   |689          |-0.448640106038448|0.367799454254972|-0.0776593039248922|0.0212759338490397|0.0730199598039357|-0.0215664603255176|0.00551293588586306|0.00371433016289403|-0.0374195960377811|-0.0196666395432079|-0.0160033965510907|-0.0160660637858097|-0.0048734327538643|-0.00654757700263541|0.00574186887214573|-0.00150087076234463|-0.0295182486311294|-0.0233162421497873|-0.0142684578471021|-0.0192239697631288|0.295698924731183|0.528117359413203|0     |0     |0     |0     |0     |0     |0     |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |False     |False      |True       |False       |True       |False      |False      |False      |False      |False       |False       |False       |False         |False         |True          |False         |True       |False      |False       |False       |True        |False       |True       |False      |False              |False              |False              |True               |False              |False              |False              |False              |False              |False              |False               |False               |False               |

`beatmaps.csv`

|id       |title                   |artist    |creator |version       |hp_drain_rate|circle_size|overall_difficulty|approach_rate|slider_multiplier|slider_tick_rate|break_points                        |status|ranked_date              |difficulty_rating|beatmap_id|
|---------|------------------------|----------|--------|--------------|-------------|-----------|------------------|-------------|-----------------|----------------|------------------------------------|------|-------------------------|-----------------|----------|
|1509063-0|Break Away (feat. RIENK)|Andromedik|Flowziee|Tachi's Hard  |3.5          |3.6        |6.2               |7.6          |1.42             |1               |['2,25849,35874', '2,136193,148977']|ranked|2024-09-05 15:44:10+00:00|3.28             |1509063   |
|1509063-1|Break Away (feat. RIENK)|Andromedik|Flowziee|Reminiscent   |5            |4.2        |9                 |9.3          |1.6              |1               |['2,25849,36130', '2,136193,151992']|ranked|2024-09-05 15:44:10+00:00|6.15             |1509063   |
|1509063-2|Break Away (feat. RIENK)|Andromedik|Flowziee|Freude's Extra|5            |3.9        |8.7               |9.1          |1.6              |1               |['2,25850,36101', '2,136193,151962']|ranked|2024-09-05 15:44:10+00:00|5.51             |1509063   |

