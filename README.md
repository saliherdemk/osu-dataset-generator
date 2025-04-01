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
| id          | time  | type   | x   | y   | hit_sound | path                 | repeat | length          | spinner_time | beatmap_id |
|------------|-------|--------|-----|-----|-----------|----------------------|--------|----------------|--------------|------------|
| 2338303-0  | 1375  | circle | 101 | 325 | 0         |                      | 0      | 0              | 0            | 2338303    |
| 2338303-0  | 4708  | circle | 434 | 240 | 0         |                      | 0      | 0              | 0            | 2338303    |
| 2338303-0  | 5541  | circle | 329 | 310 | 0         |                      | 0      | 0              | 0            | 2338303    |
| 2338303-0  | 6375  | circle | 199 | 233 | 0         |                      | 0      | 0              | 0            | 2338303    |
| 2338303-0  | 8041  | slider | 71  | 0   | 0         | P\|3:46\|28:141      | 1      | 90.2000027526856 | 0            | 2338303    |




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
Here is the final dataset format:

|unique_id|id         |time        |x       |y            |repeat      |length  |spinner_time|beatmap_id    |rms         |beat_length|meter        |slider_velocity|volume          |mfcc_1  |mfcc_2          |mfcc_3          |mfcc_4          |mfcc_5          |mfcc_6          |mfcc_7          |mfcc_8          |mfcc_9          |mfcc_10         |mfcc_11         |mfcc_12         |mfcc_13         |mfcc_14         |mfcc_15         |mfcc_16         |mfcc_17         |mfcc_18         |mfcc_19         |mfcc_20         |path_1          |path_2   |path_3   |path_4   |path_5   |path_6   |path_7   |path_8   |path_9   |path_10  |path_11  |path_12  |path_13  |path_14  |path_15  |path_16  |path_17  |path_18  |path_19  |path_20  |path_21  |path_22  |path_23  |path_24  |path_25  |path_26  |path_27  |path_28  |path_29  |path_30  |path_31  |path_32  |path_33  |path_34  |path_35  |path_36  |path_37  |path_38  |path_39  |path_40  |path_41  |path_42  |path_43  |path_44     |path_45      |path_46      |type_break    |type_circle   |type_slider   |type_spinner  |hit_sound_0   |hit_sound_1   |hit_sound_2   |hit_sound_4    |hit_sound_6    |hit_sound_8     |hit_sound_10    |hit_sound_12    |sample_set_0.0  |sample_set_1.0  |sample_set_2.0|effects_0.0   |effects_1.0   |curve_type_B  |curve_type_E  |curve_type_L  |curve_type_P|
|---------|-----------|------------|--------|-------------|------------|--------|------------|--------------|------------|-----------|-------------|---------------|----------------|--------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|------------|-------------|-------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|---------------|----------------|----------------|----------------|----------------|----------------|--------------|--------------|--------------|--------------|--------------|--------------|------------|
|1141     |2337169-0  |148161      |0.810546875|0.716883116883117|0.693147180559945|4.64918709029583|0           |2337169       |0.404625140341328|6.53764067231476|1.6094379124341|3.79500055002248|0.8             |-0.122158897523816|0.281101685020458|-0.0627068774196981|0.0767944055724958|0.0404246296746267|-0.00991939014649823|0.00334465894168866|-0.0119218272709773|-0.0124925337115102|0.0159197739086903|-0.000213638400233044|0.00367614575610943|-0.00586265478544593|-0.00127004848757321|-0.00431882103495793|-0.00269763629622559|-0.0200194969113067|-0.0106717165771179|-0.00275404970649469|-0.00775053186585654|0.876344086021505|0.733496332518338|0.924731182795699|0.76039119804401|0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0           |0            |0            |False         |False         |True          |False         |True          |False         |False         |False          |False          |False           |False           |False           |False           |False           |True          |True          |False         |False         |False         |False         |True        |
|1058     |2337169-0  |118851      |0.86328125|0.948051948051948|0           |0       |0           |2337169       |0.325842844654861|6.53764067231476|1.6094379124341|4.15103990589865|0.6             |-0.194485666529321|0.246512287603032|-0.0460036847143607|0.0701818368697515|0.058846454607051|0.00528659591116113|-0.00738276622200724|0.00687892514156417|-0.0178194872725236|-0.0116335997259892|-0.0208688371942187|-0.00727482165926707|-0.00229032128253463|0.0125883589167281|-0.00857776771558191|-0.0238352882275098|0.0286489936670856|0.0249170539179047|-0.00912164595140678|0.0166918098687615|0               |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0        |0           |0            |0            |False         |True          |False         |False         |True          |False         |False         |False          |False          |False           |False           |False           |False           |False           |True          |True          |False         |False         |True          |False         |False       |




# Merge datasets

If you collect more data later, you can merge datasets. After processing your second dataset, merge it with `merge_datasets.py`

```
python Dataset/merge_datasets.py --dataset_one=/mnt/L-HDD/dataset1 --dataset_two=/mnt/L-HDD/dataset2 --output_file=/mnt/L-HDD/merged.csv
```

Notice that `--output_file` argument is a file path not a directory. This script assumes both dataset folders have been processed and contain a `hit_objects_formatted.csv` file. It will filter and add beatmaps from the second dataset that are not present in the first.


