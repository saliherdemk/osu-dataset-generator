# Osu Dataset Generator
This repository processes beatmaps from the rhythm game called [osu!](https://osu.ppy.sh/). It analyzes, processes, reformats, and normalizes both audio and hit objects. The final dataset will serve as training data for an osu! AI beatmap generator. The goal is to create an AI that can generate proper beatmaps from just an mp3 file, supporting a variety of difficulty levels. 

# Environment Setup
This project uses the Python version specified in the `.python-version` file. You can use [pyenv](https://github.com/pyenv/pyenv) to automatically switch to the correct version.

Create virtual environment

```
python -m venv .venv
source .venv/bin/activate
```

Install packages

```
pip install -r requirements.txt
```

# Dataset Pipeline

Download your current beatmapset using [this](https://github.com/saliherdemk/osu-lazer-backup) tool. This will give you `.osz` files for your beatmapsets. 

Use `extract_osz.py` to unzip them.


```
python Dataset/pipeline/extract_osz.py --input_folder=/your_path/songs --output_folder=/your_path/extracted
```

Now you have .osu file and corresponding audio file for every beatmap that you own. Use `generate_dataset.py` script to generate your dataset.


```
python Dataset/pipeline/generate_dataset.py --input_folder=/your_path/extracted --dataset_path=/your_path/dataset
```

That will generate 3 files and one folder.

`beatmaps.csv`, `hit_objects.csv`, `timing_points.csv` and `audio` folder which contains only the song audio file.

Next, retrieve beatmap metadata and add it to the dataset. For this, you need an OAuth key from osu. Get your client id and client secret, then paste them into a `.env` file, which you will create in the base folder.

```
CLIENT_ID=your_id
CLIENT_SECRET=your_secret
```
 
```
python Dataset/pipeline/add_beatmaps_metadata.py --dataset_folder=/your_path/dataset
```

Filter ranked maps and remove old maps. (ie > 2010) 
```
python Dataset/pipeline/filter_ranked.py --dataset_folder=/your_path/dataset
```

Some of the audio files might be corrupted or not ready for processing. Fix those.

```
python Dataset/pipeline/fix_corrupted_audio.py --dataset_folder=/your_path/dataset
```

# Run Pipeline
You can run everything at once using the `run_pipeline.sh` script.

```
chmod +x /Dataset/run_pipeline.sh
```

```
./Dataset/run_pipeline.sh /your_path/songs /your_path/dataset /your_path/temp/
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
python Dataset/format_dataset.py --dataset_path=/your_path/dataset
```

Now you should have your `formatted.csv` file on your dataset folder.

`formatted.csv` should look like this:
|id       |time            |type  |x  |y  |hit_sound|path   |repeat|length          |spinner_time|new_combo|beatmap_id|beat_length     |meter|slider_velocity  |sample_set|volume|effects|difficulty_rating|frame_time      |rms        |has_hit_object|mfcc_1    |mfcc_2  |mfcc_3    |mfcc_4    |mfcc_5   |mfcc_6    |mfcc_7  |mfcc_8    |mfcc_9    |mfcc_10   |mfcc_11   |mfcc_12   |mfcc_13  |mfcc_14  |mfcc_15  |mfcc_16  |mfcc_17 |mfcc_18    |mfcc_19   |mfcc_20   |
|---------|----------------|------|---|---|---------|-------|------|----------------|------------|---------|----------|----------------|-----|-----------------|----------|------|-------|-----------------|----------------|-----------|--------------|----------|--------|----------|----------|---------|----------|--------|----------|----------|----------|----------|----------|---------|---------|---------|---------|--------|-----------|----------|----------|
|2337169-0|231             |slider|86 |114|0        |L&#124;81:47|1     |67.5000025749208|0           |True     |2337169   |689.655172413793|4    |-133.333333333333|2         |40    |0      |4.42             |220.589569160998|0.009649204|True          |-450.08582|76.15621|1.0709565 |-9.865276 |1.2259159|1.8149039 |6.453965|0.1860156 |-19.403503|-24.475204|-17.776352|-9.894934 |5.3546524|11.042467|-4.027216|-8.553526|4.964045|2.0848522  |-12.029092|-0.9998989|
|2337169-0|232.199546485261|      |   |   |         |       |      |                |            |         |2337169   |                |     |                 |          |      |       |                 |232.199546485261|0.024603726|False         |-361.93954|125.3844|-22.152828|-4.5030856|10.08624 |-10.239243|6.575365|-0.6939366|-27.928047|-23.1558  |-17.890675|-10.738436|12.190224|12.025063|-7.063714|-8.029292|3.16954 |-0.16949219|-15.290938|-6.887203 |
|2337169-0|11609           |slider|387|0  |0        |P&#124;359:12&#124;329:45|1     |58.4999973220826|0           |False    |2337169   |689.655172413793|4    |-76.9230769230769|2         |60    |0      |4.42             |11598.3673469388|0.18942297 |True          |-75.53014 |57.45023|30.394821 |44.164585 |10.207699|30.548073 |10.782623|13.519032 |-0.89644486|-1.6839523|-16.745308|-3.2124329|1.0503346|9.376907 |4.470006 |-0.87697923|-11.741819|3.6686616  |-0.110064745|-1.6746883|
|2337169-0|11609.977324263 |      |   |   |         |       |      |                |            |         |2337169   |                |     |                 |          |      |       |                 |11609.977324263 |0.19190039 |False         |-75.5213  |60.621395|31.747063 |42.687218 |10.75659 |31.396027 |10.988749|12.916475 |-0.25296926|-3.476209 |-17.38247 |-1.6582112|0.01946672|7.639636 |2.4652402|-2.2092977|-10.815493|3.794812   |-0.7236271|-2.1331735|

> **Warning:** If you have 16GB of memory or less, you might not be able to process large datasets. The `format_dataset` script includes a checkpoint system that automatically recovers progress from a specified folder. However, if you expect your processing to be interrupted multiple times, I recommend manually creating a `processed.txt` file that contains the processed `beatmap_id`s, and uncommenting the related code block in the `format_dataset` function. That would be much faster since you can use the same processed file over and over again. You can refer to the `notebooks/processed.ipynb` notebook for guidance. An alternative approach is to split your dataset into multiple parts (see `notebooks/split_dataset.ipynb`) and merge them after processing.


## Split Curve Points Into Columns
Since we stored our curve points as string in one cell. We need to spread across to new columns.


```
python Dataset/split_to_columns.py --input_file=/your_path/dataset/formatted.csv --output_file=/your_path/dataset/splitted.csv
```

Notice that both arguments take file path.

## Normalize Dataset

Before normalization, we need to get the mfcc mean and std.

```
python Dataset/get_mfcc_parameters.py --file=/your_path/dataset/splitted.csv --output_file=/your_path/dataset/mfcc.json 
```

Notice that output file is a json file. Do not delete this file. We need for denormalization step.

```
python Dataset/normalize.py --input_file=/your_path/dataset/splitted.csv --output_file=/your_path/dataset/normalized.csv --mfcc_parameters=/your_path/dataset/mfcc.json
```

You can find out more about normalization in `notebooks/normalize.ipynb` 
Here is the final dataset format:


# Merge datasets

If you collect more data later, you can merge datasets. After processing your second dataset, merge it with `merge_datasets.py`

```
python Dataset/merge_datasets.py --file_one=/your_path/dataset1.csv --file_two=/your_path/dataset2.csv --output_file=/your_path/merged.csv
```

This script assumes your files don’t have any overlapping records. It just adds the rows one after another. Be aware that it does not check for duplicate rows.


## Final Data Format

`normalized.csv`
|id       |time           |x          |y                |repeat|length|spinner_time|beatmap_id|beat_length     |meter          |slider_velocity |volume|frame_time      |rms              |mfcc_1            |mfcc_2           |mfcc_3              |mfcc_4            |mfcc_5             |mfcc_6              |mfcc_7              |mfcc_8            |mfcc_9             |mfcc_10             |mfcc_11            |mfcc_12            |mfcc_13             |mfcc_14            |mfcc_15            |mfcc_16             |mfcc_17            |mfcc_18            |mfcc_19            |mfcc_20            |path_1|path_2|path_3|path_4|path_5|path_6|path_7|path_8|path_9|path_10|path_11|path_12|path_13|path_14|path_15|path_16|path_17|path_18|path_19|path_20|path_21|path_22|path_23|path_24|path_25|path_26|path_27|path_28|path_29|path_30|path_31|path_32|path_33|path_34|path_35|path_36|path_37|path_38|path_39|path_40|path_41|path_42|path_43|path_44|path_45|path_46|type_circle|type_slider|type_slient|type_spinner|hit_sound_0.0|hit_sound_2.0|hit_sound_4.0|hit_sound_6.0|hit_sound_8.0|hit_sound_10.0|hit_sound_12.0|hit_sound_14.0|sample_set_0.0|sample_set_1.0|sample_set_2.0|sample_set_3.0|effects_0.0|effects_1.0|curve_type_B|curve_type_E|curve_type_L|curve_type_P|new_combo_False|new_combo_True|difficulty_rating_2|difficulty_rating_3|difficulty_rating_4|difficulty_rating_5|difficulty_rating_6|has_hit_object_False|has_hit_object_True|hit_sound_0|hit_sound_2|hit_sound_4|hit_sound_6|hit_sound_8|hit_sound_10|hit_sound_12|hit_sound_14|new_combo_0|new_combo_1|difficulty_rating_0|difficulty_rating_1|difficulty_rating_7|difficulty_rating_8|difficulty_rating_9|difficulty_rating_10|difficulty_rating_11|difficulty_rating_12|
|---------|---------------|-----------|-----------------|------|------|------------|----------|----------------|---------------|----------------|------|----------------|-----------------|------------------|-----------------|--------------------|------------------|-------------------|--------------------|--------------------|------------------|-------------------|--------------------|-------------------|-------------------|--------------------|-------------------|-------------------|--------------------|-------------------|-------------------|-------------------|-------------------|------|------|------|------|------|------|------|------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-----------|-----------|-----------|------------|-------------|-------------|-------------|-------------|-------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|-----------|-----------|------------|------------|------------|------------|---------------|--------------|-------------------|-------------------|-------------------|-------------------|-------------------|--------------------|-------------------|-----------|-----------|-----------|-----------|-----------|------------|------------|------------|-----------|-----------|-------------------|-------------------|-------------------|-------------------|-------------------|--------------------|--------------------|--------------------|
|2338303-0|1375           |0.197265625|0.844155844155844|0     |0     |0           |2338303   |6.03468366622796|1.6094379124341|4.71949044301739|0.65  |1369.97732426304|0.227322266814544|-0.158797591336191|0.122393059738145|0.000758968444056121|0.0569715259216494|0.00532830958553307|0.0116146017330744  |-0.00229196168225229|0.0195894295288504|-0.0187735196905712|-0.00930820801033577|-0.0293281416539657|0.00150602417810011|-0.00538465828127316|-0.0105567209096735|-0.0152564619246108|-0.00628491149145721|-0.0332661706220421|-0.0288390102254618|-0.0275988867652196|-0.0159905715586988|0     |0     |0     |0     |0     |0     |0     |0     |0     |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |True       |False      |False      |False       |True         |False        |False        |False        |False        |False         |False         |False         |False         |False         |True          |False         |True       |False      |False       |True        |False       |False       |False          |True          |False              |False              |True               |False              |False              |False               |True               |False      |False      |False      |False      |False      |False       |False       |False       |False      |False      |False              |False              |False              |False              |False              |False               |False               |False               |
|2338303-0|1381.5873015873|0          |0                |0     |0     |0           |2338303   |0               |0              |0               |0     |1381.5873015873 |0.308761735343268|-0.045539661815015|0.132512086888443|-0.0285476604582358 |0.0361374037977998|0.0168191119608961 |-0.00308032959326649|-0.0180461923658881 |0.015524353627074 |-0.0122573471151447|-0.0132275182010809 |-0.0219876085224041|0.0059121886865447 |-0.019681413112806  |-0.0273487891059135|-0.0132833818228793|-0.00950979807716372|-0.0337777848112838|-0.0255938358088564|-0.025102169938417 |-0.013355427598629 |0     |0     |0     |0     |0     |0     |0     |0     |0     |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |False      |False      |True       |False       |True         |False        |False        |False        |False        |False         |False         |False         |True          |False         |False         |False         |True       |False      |False       |True        |False       |False       |True           |False         |False              |False              |True               |False              |False              |True                |False              |False      |False      |False      |False      |False      |False       |False       |False       |False      |False      |False              |False              |False              |False              |False              |False               |False               |False               |

`beatmaps.csv`

|id       |title                |artist                |creator |version    |hp_drain_rate|circle_size|overall_difficulty|approach_rate|slider_multiplier|slider_tick_rate|break_points                        |status|ranked_date              |difficulty_rating|beatmap_id|
|---------|---------------------|----------------------|--------|-----------|-------------|-----------|------------------|-------------|-----------------|----------------|------------------------------------|------|-------------------------|-----------------|----------|
|2338303-0|Shinsekai Koukyougaku|Sayonara Ponytail     |SUISEI69|Insane     |5            |4          |7                 |8            |1.64             |2               |['2,89075,92291', '2,198241,210625']|ranked|2025-03-23 17:05:50+00:00|4.25             |2338303   |
|2340739-0|LEMON MELON COOKIE   |TAK feat. Hatsune Miku|Andrea  |Amats' Hard|4            |3.4        |6.5               |7.5          |1.4              |1               |['2,35196,38171']                   |ranked|2025-03-25 00:43:22+00:00|3.53             |2340739   |

