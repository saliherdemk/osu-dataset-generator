# osu! Beatmap Processing and AI Generator

## Environment Setup
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

## Dataset Pipeline

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

### Run Pipeline
You can run everything at once using the `run_pipeline.sh` script.

```
chmod +x /Dataset/run_pipeline.sh
```

```
./Dataset/run_pipeline.sh /your_path/songs /your_path/dataset /your_path/temp/
```


### Formatting

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


Additionally, I calculated `duration` which mean how many milliseconds it takes to complete one slide of the slider or  how many ms to complete the spinner. [sliders](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29#sliders)


```
python Dataset/format_dataset.py --dataset_path=/your_path/dataset
```

Now you should have `formatted.csv` file which should look like this:


|id     |time|type  |x  |y  |hit_sound|path                   |repeat|spinner_time|new_combo|slider_velocity|sample_set|volume|effects|difficulty_rating|meter|beat_length     |mapper_id|beatmap_id|duration|
|-------|----|------|---|---|---------|-----------------------|------|------------|---------|---------------|----------|------|-------|-----------------|-----|----------------|---------|----------|--------|
|18779-0|3221|circle|400|88 |0        |E&#124;                     |0     |0           |False    |1.4            |1         |76    |0      |2.88             |4    |368.098159509202|73453    |18779     |0       |
|18779-0|3405|slider|400|88 |0        |B&#124;320:56&#124;248:80&#124;208:168|1     |0           |False    |1.4            |1         |76    |0      |2.88             |4    |368.098159509202|73453    |18779     |552     |
|18779-0|4141|circle|219|146|4        |E&#124;                     |0     |0           |True     |1.4            |1         |76    |0      |2.88             |4    |368.098159509202|73453    |18779     |0       |

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
| duration        | int64    |

# Timing Model

I used CRNN to predict hit object times and types based on audio file and provided difficulty rating.

## Model Architecture
TODO

## Training

You need `audio` folder which contains audio files and `formatted.csv` file for that. Assuming both locating in the parent folder `dataset`, you can train model on that dataset with

``` 
python Architecture/main.py --mode=train --dataset_folder=/your_path/dataset --save_to=/your_checkpoint_path --load_from=/your_last_checkpoint_file.pt 
```

`main.py` arguments:

```
parser.add_argument("--dataset_folder", default=None)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--load_from", default=None)
parser.add_argument("--save_to", default=None)
parser.add_argument("--mode", default="train", choices=["train", "predict"])
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--audio_path", default=None)
parser.add_argument("--diff_rating", default=None, type=float)
```

## Prediction

```
python Architecture/main.py --mode=predict --load_from=/your_last_checkpoint_file.pt --audio_path=/home/saliherdemk/try_dataset/audio/18779.mp3 --diff_rating=2.88 --save_to=/home/saliherdemk/try_dataset/checkpoint/
```




