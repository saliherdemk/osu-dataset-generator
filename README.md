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
|id       |x          |y                |repeat           |slider_time     |spinner_time|beat_length     |meter          |slider_velocity  |volume|rms        |mfcc_1           |mfcc_2           |mfcc_3           |mfcc_4           |mfcc_5           |mfcc_6            |mfcc_7           |mfcc_8            |mfcc_9            |mfcc_10          |mfcc_11             |mfcc_12          |mfcc_13          |mfcc_14          |mfcc_15           |mfcc_16           |mfcc_17          |mfcc_18           |mfcc_19           |mfcc_20          |path_1     |path_2           |path_3|path_4|path_5|path_6|path_7|path_8|path_9|path_10|path_11|path_12|path_13|path_14|path_15|path_16|path_17|path_18|path_19|path_20|path_21|path_22|path_23|path_24|path_25|path_26|path_27|path_28|path_29|path_30|path_31|path_32|path_33|path_34|path_35|path_36|path_37|path_38|path_39|path_40|path_41|path_42|path_43|path_44|path_45|path_46|delta_time        |type_circle|type_slider|type_slient|type_spinner|hit_sound_0.0|hit_sound_2.0|hit_sound_4.0|hit_sound_6.0|hit_sound_8.0|hit_sound_10.0|hit_sound_12.0|hit_sound_14.0|sample_set_0.0|sample_set_1.0|sample_set_2.0|sample_set_3.0|effects_0.0|effects_1.0|curve_type_B|curve_type_E|curve_type_L|curve_type_P|new_combo_False|new_combo_True|difficulty_rating_3|difficulty_rating_4|difficulty_rating_5|difficulty_rating_6|has_hit_object_False|has_hit_object_True|difficulty_rating_1|difficulty_rating_2|difficulty_rating_7|
|---------|-----------|-----------------|-----------------|----------------|------------|----------------|---------------|-----------------|------|-----------|-----------------|-----------------|-----------------|-----------------|-----------------|------------------|-----------------|------------------|------------------|-----------------|--------------------|-----------------|-----------------|-----------------|------------------|------------------|-----------------|------------------|------------------|-----------------|-----------|-----------------|------|------|------|------|------|------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|------------------|-----------|-----------|-----------|------------|-------------|-------------|-------------|-------------|-------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|-----------|-----------|------------|------------|------------|------------|---------------|--------------|-------------------|-------------------|-------------------|-------------------|--------------------|-------------------|-------------------|-------------------|-------------------|
|2331782-0|0.744140625|0.290909090909091|0                |0               |0           |5.82327856570727|1.6094379124341|1.02961941718116 |0.6   |0.015472304|-3.13867877270378|-1.78795330649548|-2.99251259370535|-4.22054533963214|-1.52301737888156|-0.184947125876565|0.618431484044184|0.559024994149907 |-0.819175182235321|-1.64478656570695|0.522344414917874   |0.654094634459249|3.62089990157612 |1.97693024790966 |-1.03080426642751 |-1.52487041046825 |-0.41929542419157|-1.24708343705265 |-0.126184237066283|1.02641195005551 |0          |0                |0     |0     |0     |0     |0     |0     |0     |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |-0.132616928482561|True       |False      |False      |False       |True         |False        |False        |False        |False        |False         |False         |False         |False         |True          |False         |False         |True       |False      |False       |True        |False       |False       |False          |True          |False              |False              |True               |False              |False               |True               |False              |False              |False              |
|2331782-0|0.765625   |0.264935064935065|0                |0               |0           |5.82327856570727|1.6094379124341|1.02961941718116 |0.6   |0.030624755|-3.02598228100316|-1.0641364169191 |-2.12829220751379|-4.24730021528174|0.401263310915646|2.30021123049946  |3.93215250574334 |-0.719562669614891|-2.12497437732127 |1.16952385344091 |-0.00699602418341917|1.56035559530526 |0.547105051873934|0.196285767390237|-0.822949805232656|-0.850774268677755|-1.2835569419266 |-1.44944065448212 |-0.390890211771191|-1.085754603591  |0          |0                |0     |0     |0     |0     |0     |0     |0     |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |-0.105859168243812|True       |False      |False      |False       |True         |False        |False        |False        |False        |False         |False         |False         |False         |True          |False         |False         |True       |False      |False       |True        |False       |False       |True           |False         |False              |False              |True               |False              |False               |True               |False              |False              |False              |
|2331782-0|0.8203125  |0.244155844155844|0.693147180559945|5.82327860374141|0           |5.82327856570727|1.6094379124341|0.854415328156068|0.25  |0.056320217|-1.85856672386295|1.03680895315481 |-1.80637827807063|-3.71317325986711|-2.1589489049876 |-1.10449466944275 |-2.58264091391181|-4.41632986646031 |-0.872775266304271|1.70754671412528 |1.01689306051805    |-1.16459834182374|-1.69139647699854|0.223144061088371|0.842855518257226 |-0.618032254875412|-2.62901181332602|-0.626957128847063|0.541741102518245 |0.988206114305976|0.556640625|0.215584415584416|0     |0     |0     |0     |0     |0     |0     |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0      |0.0337889965058585|False      |True       |False      |False       |True         |False        |False        |False        |False        |False         |False         |False         |False         |False         |True          |False         |True       |False      |True        |False       |False       |False       |False          |True          |False              |False              |True               |False              |False               |True               |False              |False              |False              |

`beatmaps.csv`

|id       |title                |artist                |creator |version    |hp_drain_rate|circle_size|overall_difficulty|approach_rate|slider_multiplier|slider_tick_rate|break_points                        |status|ranked_date              |difficulty_rating|beatmap_id|
|---------|---------------------|----------------------|--------|-----------|-------------|-----------|------------------|-------------|-----------------|----------------|------------------------------------|------|-------------------------|-----------------|----------|
|2338303-0|Shinsekai Koukyougaku|Sayonara Ponytail     |SUISEI69|Insane     |5            |4          |7                 |8            |1.64             |2               |['2,89075,92291', '2,198241,210625']|ranked|2025-03-23 17:05:50+00:00|4.25             |2338303   |
|2340739-0|LEMON MELON COOKIE   |TAK feat. Hatsune Miku|Andrea  |Amats' Hard|4            |3.4        |6.5               |7.5          |1.4              |1               |['2,35196,38171']                   |ranked|2025-03-25 00:43:22+00:00|3.53             |2340739   |

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
