import pandas as pd

df = pd.read_csv("/home/saliherdemk/osu-dataset/encoded.csv")


def last_hit_object_index(tokenized_str):
    tokens = tokenized_str.split(",")
    try:
        last_idx = len(tokens) - 1 - tokens[::-1].index("<hit_object_start>")
        return ",".join(tokens[last_idx:])
    except ValueError:
        return ""


df["last_hit_object"] = df["tokenized"].apply(last_hit_object_index)
df["prev_last_hit_object"] = df.groupby("beatmap_id")["last_hit_object"].shift(1)
prev = df["prev_last_hit_object"].fillna("")
df["tokenized_with_overlap"] = (prev + "," + df["tokenized"]).str.strip(",")
df.to_csv("/home/saliherdemk/osu-dataset/extended.csv", index=False)
