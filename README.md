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


# Formatting

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

Now you should have `formatted` folder which contains `formatted.csv` file and `mels` folder. 

`formatted.csv` should look like this:

|id       |time|type   |x  |y  |hit_sound|path             |repeat|spinner_time|new_combo|slider_velocity|sample_set|volume|effects|difficulty_rating|meter|beat_length    |mapper_id|beatmap_id|tick|delta_time|
|---------|----|-------|---|---|---------|-----------------|------|------------|---------|---------------|----------|------|-------|-----------------|-----|---------------|---------|----------|----|----------|
|2182049-0|2644|spinner|256|192|0        |E&#124;               |0     |4846        |True     |2.4            |2         |30    |0      |3.08             |4    |550.45871559633|8147142  |2182049   |16  |0         |
|2182049-0|5398|slider |339|153|0        |P&#124;365:129&#124;468:175|1     |0           |True     |1.44           |2         |30    |0      |3.08             |4    |550.45871559633|8147142  |2182049   |4   |2754      |
|2182049-0|6223|slider |395|213|0  |P&#124;368:236&#124;265:190|1  |0   |False|1.44|2  |30 |0  |3.08|4  |550.45871559633|8147142|2182049|4  |825 |
|2182049-0|7324|circle |168|71 |0  |E&#124;               |0  |0   |True |1.44|2  |30 |0  |3.08|4  |550.45871559633|8147142|2182049|0  |1101|


| Field             | Type     |
|-------------------|----------|
| id                | string   |
| time              | float64  |
| type              | string   |
| x                 | int16    |
| y                 | int16    |
| hit_sound         | int8     |
| path              | string   |
| repeat            | int16    |
| spinner_time      | int32    |
| new_combo         | bool     |
| slider_velocity   | float64  |
| sample_set        | int8     |
| volume            | int8     |
| effects           | int8     |
| difficulty_rating | float16  |
| meter             | int8     |
| beat_length       | float64  |
| mapper_id         | int64    |
| beatmap_id        | int64    |
| tick              | int64    |
| delta_time        | int64    |

`delta_time` is the difference between the hit object’s time and the previous hit object’s time. `tick` represents how many ticks it takes to complete the slider or spinner. The others are self-explanatory.

# Tokenizing

You can find the vocabulary in the `Tokenizer/vocab/` folder. Most of the tokenization is straightforward. I applied some normalization while tokenizing.

* `x` and `y` values are snapped to the 32 px grid.
* `volume` column is snapped to a multiple of 10.
* `delta_time` has a range from 0 to 2000. Larger delta_time values are represented as a combination of these values. (Example: 4500 → 2000, 2000, 500)
* `slider_velocity` has a precision of 0.1.
* `tick` has a range from 0 to 50. Larger values are represented as a combination of these tokens.
* `repeat` has a range from 0 to 30. Larger values are represented as a combination of these tokens. (didn't like that. repeat should be single token. UPDATE: I hoper there will be practiacal limition of being ranked but appreantly there is not. Beatmap `1862270` has 96 repeat in one slider and it's ranked. Also in fine tuning people might one to create tech-based map so repeat will be stay as it is.)


