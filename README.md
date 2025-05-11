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

|id       |time |beat_length      |meter|sample_set|volume|uninherited|effects|beatmap_id|
|---------|-----|-----------------|-----|----------|------|-----------|-------|----------|
|2331782-0|1173 |337.078651685393 |4    |2         |25    |1          |0      |2331782   |
|2331782-0|1173 |-133.333333333333|4    |2         |25    |0          |0      |2331782   |
|2331782-0|9262 |-200             |4    |2         |30    |0          |0      |2331782   |
|2331782-0|10611|-200             |4    |2         |40    |0          |0      |2331782   |



example `hit_objects.csv`
|id       |time |type             |x  |y  |hit_sound|path|repeat|length |spinner_time|new_combo|beatmap_id|
|---------|-----|-----------------|---|---|---------|----|------|-------|------------|---------|----------|
|2331782-0|161  |circle           |381|112|0        |    |0     |0      |0           |True     |2331782   |
|2331782-0|498  |circle           |392|102|0        |    |0     |0      |0           |False    |2331782   |
|2331782-0|835  |circle           |406|96 |0        |    |0     |0      |0           |False    |2331782   |
|2331782-0|1173 |slider           |420|94 |0        |B&#124;285:83|1     |135.000005149842|0           |True     |2331782   |





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
|id       |time           |type   |x  |y  |hit_sound|path    |repeat|slider_time     |spinner_time|new_combo|beatmap_id|beat_length     |meter|slider_velocity|sample_set|volume|effects|difficulty_rating|frame_time      |rms        |has_hit_object|mfcc_1   |mfcc_2    |mfcc_3    |mfcc_4    |mfcc_5    |mfcc_6    |mfcc_7    |mfcc_8   |mfcc_9    |mfcc_10   |mfcc_11  |mfcc_12  |mfcc_13   |mfcc_14  |mfcc_15   |mfcc_16   |mfcc_17   |mfcc_18   |mfcc_19   |mfcc_20  |
|---------|---------------|-------|---|---|---------|--------|------|----------------|------------|---------|----------|----------------|-----|---------------|----------|------|-------|-----------------|----------------|-----------|--------------|---------|----------|----------|----------|----------|----------|----------|---------|----------|----------|---------|---------|----------|---------|----------|----------|----------|----------|----------|---------|
|2331782-0|150.92970521542|slient |   |   |         |        |      |                |            |False    |2331782   |                |     |               |          |      |       |                 |150.92970521542 |0.011138746|False         |-489.9509|14.7284975|-34.745544|-29.308521|1.2013087 |16.088156 |15.381813 |6.9820266|-8.003919 |-14.232279|-8.602083|3.0424132|15.556789 |8.736643 |-7.5157433|-8.436203 |-1.9449265|-1.2086651|2.1680002 |9.338654 |
|2331782-0|161            |circle |381|112|0        |        |0     |0               |0           |True     |2331782   |337.078651685393|4    |1.8            |1         |60    |0      |5.16             |162.539682539683|0.015472304|True          |-410.7289|29.359741 |-72.14767 |-49.475258|-13.537962|7.804775  |6.4304905 |10.900503|-10.597628|-9.159056 |0.8171804|6.1773024|23.0276   |14.404533|-10.102219|-9.483953 |-4.5893626|-6.520191 |-2.8025599|10.272303|
|2331782-0|1173           |slider |420|94 |0        |B&#124;285:83|1     |337.078664543925|0           |True     |2331782   |337.078651685393|4    |1.35           |2         |25    |0      |5.16             |1172.60770975057|0.056320217|True          |-257.8448|114.471725|-43.56835 |-40.368195|-21.538124|-2.3357728|-25.465197|-35.42932|-11.143511|18.507385 |5.006461 |-8.084826|-16.835087|1.6332331|3.3523166 |-2.6297388|-21.69119 |-1.7446318|2.6835296 |10.004623|
|2331782-0|176537         |spinner|256|192|2        |        |0     |0               |177801      |True     |2331782   |337.078651685393|4    |1.44           |2         |50    |0      |5.16             |176541.315192744|0.31305823 |True          |10.904665|89.47315  |-20.447561|36.242958 |9.44141   |24.198387 |20.20139  |4.357092 |12.751471 |6.5978317 |-8.439117|2.2036352|-1.9157301|2.8436902|5.3867397 |3.3426673 |-3.0061734|5.6394205 |8.69149   |9.552832 |



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
python Dataset/get_dataset_metadata.py --input_file=/your_path/dataset/splitted.csv --output_folder=/your_path/dataset/dataset_metadata 
```

Notice that output path is folder.

```
python Dataset/normalize.py --input_file=/your_path/dataset/splitted.csv --output_file=/your_path/dataset/normalized.csv --dataset_metadata=/your_path/dataset/dataset_metadata
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
|id       |x          |y                 |repeat           |slider_time     |spinner_time|beat_length     |meter          |slider_velocity |volume|rms       |mfcc_1            |mfcc_2            |mfcc_3           |mfcc_4           |mfcc_5           |mfcc_6           |mfcc_7           |mfcc_8           |mfcc_9            |mfcc_10          |mfcc_11             |mfcc_12           |mfcc_13           |mfcc_14           |mfcc_15          |mfcc_16           |mfcc_17             |mfcc_18           |mfcc_19           |mfcc_20           |path_1     |path_2           |path_3     |path_4           |path_5     |path_6           |path_7|path_8|path_9|path_10|path_11|path_12|path_13|path_14|path_15|path_16|path_17|path_18|path_19|path_20|path_21|path_22|path_23|path_24|path_25|path_26|path_27|path_28|path_29|path_30|path_31|path_32|path_33|path_34|path_35|path_36|path_37|path_38|path_39|path_40|path_41|path_42|path_43|path_44|path_45|path_46|path_47|path_48|path_49|path_50|delta_time        |type_circle|type_slient|type_slider|type_spinner|hit_sound_0.0|hit_sound_2.0|hit_sound_4.0|hit_sound_6.0|hit_sound_8.0|hit_sound_10.0|hit_sound_12.0|hit_sound_14.0|sample_set_0.0|sample_set_1.0|sample_set_2.0|sample_set_3.0|effects_0.0|effects_1.0|curve_type_B|curve_type_E|curve_type_L|curve_type_P|new_combo_True|new_combo_False|difficulty_rating_1|difficulty_rating_2|difficulty_rating_3|difficulty_rating_4|difficulty_rating_5|difficulty_rating_6|has_hit_object_True|has_hit_object_False|
|---------|-----------|------------------|-----------------|----------------|------------|----------------|---------------|----------------|------|----------|------------------|------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|------------------|-----------------|--------------------|------------------|------------------|------------------|-----------------|------------------|--------------------|------------------|------------------|------------------|-----------|-----------------|-----------|-----------------|-----------|-----------------|------|------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|------------------|-----------|-----------|-----------|------------|-------------|-------------|-------------|-------------|-------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|-----------|-----------|------------|------------|------------|------------|--------------|---------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|--------------------|
|2325539-0|0.798828125|0.555844155844156 |0.693147180559945|5.81213849929382|0           |6.50378904699771|1.6094379124341|1.22377543162212|0.5   |0.2635273 |0.149784051864111 |-0.303519879825799|0.139899609246378|0.346444967761407|0.688888145131792|1.39455852719448 |2.00383936496145 |0.983918875969992|-0.170871856824529|-1.06526168904423|-0.00410263484842214|-0.305409107395828|0.0861982741370003|-0.481311726011105|0.296880959587248|-0.518321742765173|-0.00791783735106407|0.123761030055475 |0.326688187648984 |-0.415137086935317|0.740234375|0.415584415584416|0.740234375|0.415584415584416|0.779296875|0.275324675324675|0     |0     |0     |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |-0.43554602432419 |False      |False      |True       |False       |True         |False        |False        |False        |False        |False         |False         |False         |False         |False         |True          |False         |True       |False      |True        |False       |False       |False       |False         |True           |False              |True               |False              |False              |False              |False              |True               |False               |
|2325539-0|0          |0                 |0                |0               |0           |0               |0              |0               |0     |0.2494192 |-0.189517939383337|-0.262027084467381|0.549969707433081|0.505964179864876|1.5234480828367  |0.58344330221599 |0.252767419818538|1.53371591817382 |-0.538906511669012|2.1013757847164  |1.57096238261481    |1.08792754371276  |0.787420000118951 |0.426278957352101 |0.819182540479132|0.663045804480977 |0.632833630588632   |0.201878259936323 |-0.7360185609826  |-1.36363967014395 |0          |0                |0          |0                |0          |0                |0     |0     |0     |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0                 |False      |True       |False      |False       |True         |False        |False        |False        |False        |False         |False         |False         |True          |False         |False         |False         |True       |False      |False       |True        |False       |False       |False         |True           |False              |True               |False              |False              |False              |False              |False              |True                |
|2325539-0|0.912109375|0.0987012987012987|0.693147180559945|5.81213849929382|0           |6.50378904699771|1.6094379124341|1.22377543162212|0.5   |0.24648027|-0.116445571494083|0.046265451850276 |0.308672525684157|0.197716242522109|1.51742346386524 |0.671405474054456|0.491920258889319|1.24488319413219 |-1.2886889114066  |1.71902899811639 |1.01475075931738    |0.968123579858117 |0.701403561254422 |0.714767777913869 |0.90438507737092 |0.463609489534124 |0.813182103761404   |-0.186383483938203|-0.930946656989241|-0.544804195068897|0.970703125|0.238961038961039|0.970703125|0.238961038961039|0.931640625|0.379220779220779|0     |0     |0     |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0.0150390331266696|False      |False      |True       |False       |True         |False        |False        |False        |False        |False         |False         |False         |False         |False         |True          |False         |True       |False      |True        |False       |False       |False       |False         |True           |False              |True               |False              |False              |False              |False              |True               |False               |

`beatmaps.csv`
|id       |title               |artist  |creator|version                                          |hp_drain_rate|circle_size|overall_difficulty|approach_rate|slider_multiplier|slider_tick_rate|break_points|status|ranked_date              |difficulty_rating|beatmap_id|
|---------|--------------------|--------|-------|-------------------------------------------------|-------------|-----------|------------------|-------------|-----------------|----------------|------------|------|-------------------------|-----------------|----------|
|2331782-0|Colorful -season 03-|ClariS  |Kanui  |New Colors                                       |6            |4          |8                 |9            |1.8              |2               |[]          |ranked|2025-03-12 18:01:05+00:00|5.16             |2331782   |
|2325539-0|Fanera (Cut Ver.)   |MellSher|Arfi   |Normal                                           |3            |3.3        |4                 |5            |2.4              |1               |[]          |ranked|2025-03-22 06:02:09+00:00|2.42             |2325539   |
|2335535-0|Tensei              |Imperial Circus Dead Decadence|Ratarok|Celestial                                        |5            |4          |8.5               |9.4          |3                |1               |['2,213830,223377']|ranked|2025-03-23 13:44:37+00:00|6.06             |2335535   |

## Denormalize

After getting the model output (which contains only one beatmap -if not, see `split_grouped.ipynb`), denormalize it.
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
