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
If you have tons of betamaps you might wanna filter more strategically.
```
python Dataset/pipeline/filter_ranked.py --dataset/folder=/mnt/L-HDD/Public/ranked --min_ranked_date=2015-01-01 --excluded_diffs=0,8,9,10,11,12
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

example `timing_points.csv`

|id      |time |beat_length     |meter|sample_set|volume|uninherited|effects|beatmap_id|
|--------|-----|----------------|-----|----------|------|-----------|-------|----------|
|299224-0|141  |666.666666666667|4    |2         |30    |1          |0      |299224    |
|299224-0|21474|-100            |4    |2         |50    |0          |0      |299224    |
|299224-0|23974|-100            |4    |2         |30    |0          |0      |299224    |
|299224-0|24141|-100            |4    |2         |50    |0          |0      |299224    |



example `hit_objects.df`
|id      |time |type            |x  |y  |hit_sound|path|repeat|length|spinner_time|new_combo|beatmap_id|
|--------|-----|----------------|---|---|---------|----|------|------|------------|---------|----------|
|299224-0|21474|slider          |396|120|0        |L&#124;392:180|1     |60    |0           |True     |299224    |
|299224-0|21807|slider          |316|196|0        |L&#124;312:136|1     |60    |0           |False    |299224    |
|299224-0|22141|slider          |244|96 |0        |P&#124;188:120&#124;132:104|1     |120   |0           |False    |299224    |
|299224-0|22641|circle          |84 |44 |0        |    |0     |0     |0           |False    |299224    |





You can read what these attributes represent on the [osu wiki!](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29)

We need to get the current timing attributes for each hit object. To do this, we must find the latest timing point before the hit object and extract attributes from there. For uninherited timing points, `beat_length` represents `ms_per_beat`, whereas for inherited ones, it represents the `slider multiplier`. More than one timing point may affect the same hit object, so we need to create columns for each possible timing point.

For each row, we need `beat_length`, `meter`, `slider_velocity`, `sample_set`, `volume`, `effects`
values. Also we need the corresponding `MFCC` and `RMS` values for that time which we will extract from the audio file.

Additionally, I calculated `slider_time` which mean how many milliseconds it takes to complete one slide of the slider. [sliders](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29#sliders)


```
python Dataset/format_dataset.py --dataset_path=/your_path/dataset
```

Now you should have your `formatted.csv` file on your dataset folder.

`formatted.csv` should look like this:
|id      |time |type            |x  |y  |hit_sound|path|repeat|slider_time|spinner_time|new_combo|beatmap_id|beat_length     |meter|slider_velocity|sample_set|volume|effects|difficulty_rating|frame_time|rms|has_hit_object|mfcc_1    |mfcc_2|mfcc_3|mfcc_4|mfcc_5|mfcc_6|mfcc_7|mfcc_8|mfcc_9|mfcc_10|mfcc_11|mfcc_12|mfcc_13|mfcc_14|mfcc_15|mfcc_16|mfcc_17|mfcc_18|mfcc_19|mfcc_20|
|--------|-----|----------------|---|---|---------|----|------|-----------|------------|---------|----------|----------------|-----|---------------|----------|------|-------|-----------------|----------|---|--------------|----------|------|------|------|------|------|------|------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|299224-0|0    |slient          |   |   |         |    |      |           |            |False    |299224    |                |     |               |          |      |       |                 |0         |0  |False         |-488.14407|0     |0     |0     |0     |0     |0     |0     |0     |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |
|299224-0|29474|slider          |356|256|0        |P&#124;368:224&#124;368:192|1     |166.666666666667|0           |True     |299224    |666.666666666667|4    |2.4            |2         |50    |0      |3.23             |29477.7324263039|0.20539671|True          |-184.05023|97.440346|56.573364|34.041973|9.781199|13.510959|8.379482|6.3559504|19.975037|16.339458|-9.8904295|2.5457358|-2.0569942|-2.2656868|6.2631955|7.36822|-6.9803486|-1.5625559|-10.483891|-8.873984|
|299224-0|30641|circle          |60 |284|0        |    |0     |0          |0           |False    |299224    |666.666666666667|4    |2.4            |2         |50    |0      |3.23             |30638.7301587302|0.067649454|True          |-255.90175|115.20779|10.89217|37.412014|16.151308|-1.2708448|0.051377535|-3.835034|-12.903887|-2.4620976|-7.775306|-6.1967125|0.40131688|-1.1324121|-7.8088913|-4.0419407|-4.0880303|-5.5697174|-13.769978|-14.368175|


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
|id      |x      |y                |repeat           |slider_time     |spinner_time|beat_length     |meter          |slider_velocity |volume|rms        |mfcc_1            |mfcc_2            |mfcc_3           |mfcc_4            |mfcc_5            |mfcc_6            |mfcc_7           |mfcc_8           |mfcc_9            |mfcc_10          |mfcc_11          |mfcc_12           |mfcc_13           |mfcc_14          |mfcc_15           |mfcc_16           |mfcc_17           |mfcc_18           |mfcc_19           |mfcc_20          |path_1 |path_2           |path_3|path_4|path_5|path_6|delta_time        |type_circle|type_slider|type_slient|hit_sound_0.0|hit_sound_2.0|hit_sound_4.0|hit_sound_8.0|sample_set_0.0|sample_set_1.0|sample_set_2.0|effects_0.0|effects_1.0|curve_type_B|curve_type_E|curve_type_L|curve_type_P|new_combo_False|new_combo_True|difficulty_rating_2|difficulty_rating_3|has_hit_object_False|has_hit_object_True|type_spinner|hit_sound_6.0|hit_sound_10.0|hit_sound_12.0|hit_sound_14.0|sample_set_3.0|new_combo_0|new_combo_1|difficulty_rating_1|difficulty_rating_4|difficulty_rating_5|difficulty_rating_6|difficulty_rating_7|
|--------|-------|-----------------|-----------------|----------------|------------|----------------|---------------|----------------|------|-----------|------------------|------------------|-----------------|------------------|------------------|------------------|-----------------|-----------------|------------------|-----------------|-----------------|------------------|------------------|-----------------|------------------|------------------|------------------|------------------|------------------|-----------------|-------|-----------------|------|------|------|------|------------------|-----------|-----------|-----------|-------------|-------------|-------------|-------------|--------------|--------------|--------------|-----------|-----------|------------|------------|------------|------------|---------------|--------------|-------------------|-------------------|--------------------|-------------------|------------|-------------|--------------|--------------|--------------|--------------|-----------|-----------|-------------------|-------------------|-------------------|-------------------|-------------------|
|299224-0|0.21875|0.415584415584416|0                |0               |0           |6.50378904699771|1.6094379124341|1.22377543162212|0.5   |0.108899444|0.464015170289692 |-0.456772498936307|-1.43262514243992|-0.44451375494077 |-0.502870170243623|-0.781532265662641|-1.30111541709038|-1.02463018430025|-0.780260833704907|-1.75940074889989|-1.50066221323842|-0.920794358811875|-0.974116323044256|-1.00980110635869|-0.864295778535114|-0.3885000371316  |0.0488379471433866|-0.300969954178171|-0.266516635720011|0.488097596836826|0      |0                |0     |0     |0     |0     |-0.140429413223907|True       |False      |False      |True         |False        |False        |False        |False         |False         |True          |True       |False      |False       |True        |False       |False       |True           |False         |False              |True               |False               |True               |False       |False        |False         |False         |False         |False         |False      |False      |False              |False              |False              |False              |False              |
|299224-0|0.34375|0.301298701298701|0.693147180559945|5.12197788143163|0           |6.50378904699771|1.6094379124341|1.22377543162212|0.5   |0.10533506 |-0.147290934709501|-0.499214872756171|0.739948861097528|0.0173230983377549|-0.540627585006295|-0.148648599290585|-1.13930624015567|-1.85324737570275|-1.89837108218152 |-1.74339814377302|-1.44575193482004|-1.30034108017446 |-1.62862212601509 |-2.00563977899765|0.162840370350035 |-0.890428361532749|-1.29601170624392 |-1.32052327916373 |-0.616506016498151|-1.40596096592008|0.46875|0.290909090909091|0     |0     |0     |0     |0.157616879654477 |False      |True       |False      |True         |False        |False        |False        |False         |False         |True          |True       |False      |False       |False       |True        |False       |True           |False         |False              |True               |False               |True               |False       |False        |False         |False         |False         |False         |False      |False      |False              |False              |False              |False              |False              |
|299224-0|0      |0                |0                |0               |0           |0               |0              |0               |0     |0.13498214 |0.972897238986692 |-1.57630274732654 |0.136937390602343|-0.539207295065081|-1.02180874398952 |-0.572595438147346|-1.4288444800539 |-1.60326905973004|-1.30254440891916 |-1.48807719657899|-1.36455056873918|-0.651199286157936|-0.958270010476752|-3.20235583272965|-1.43755323069913 |-0.464323312994602|-1.07367699764056 |0.323400604217794 |0.631247884921445 |-1.33565532309219|0      |0                |0     |0     |0     |0     |0                 |False      |False      |True       |True         |False        |False        |False        |True          |False         |False         |True       |False      |False       |True        |False       |False       |True           |False         |False              |True               |True                |False              |False       |False        |False         |False         |False         |False         |False      |False      |False              |False              |False              |False              |False              |


`beatmaps.csv`

|id       |title                |artist                |creator |version    |hp_drain_rate|circle_size|overall_difficulty|approach_rate|slider_multiplier|slider_tick_rate|break_points                        |status|ranked_date              |difficulty_rating|beatmap_id|
|---------|---------------------|----------------------|--------|-----------|-------------|-----------|------------------|-------------|-----------------|----------------|------------------------------------|------|-------------------------|-----------------|----------|
|2338303-0|Shinsekai Koukyougaku|Sayonara Ponytail     |SUISEI69|Insane     |5            |4          |7                 |8            |1.64             |2               |['2,89075,92291', '2,198241,210625']|ranked|2025-03-23 17:05:50+00:00|4.25             |2338303   |
|2340739-0|LEMON MELON COOKIE   |TAK feat. Hatsune Miku|Andrea  |Amats' Hard|4            |3.4        |6.5               |7.5          |1.4              |1               |['2,35196,38171']                   |ranked|2025-03-25 00:43:22+00:00|3.53             |2340739   |

## Denormalize

After getting the model output, denormalize it.
```
python Dataset/denormalize.py --input_file=/your_path/normalized.csv --output_file=/your_path/denormalized.csv --audio_path=/your_path/audio.mp3 
```

## Generate .osu file

```
python Dataset/generate_file.py --input_file=/your_path/denormalized.csv --output_file=/your_path/song.osu
```
This will create an `.osu` file with the following default parameters:

```
osu file format v14

[General]
AudioFilename: audio.mp3

[Difficulty]
SliderMultiplier:1
SliderTickRate:2
```

Since slider velocities are relative to SliderMultiplier = 1, do not change this value. You can tweak the other values if you want.
