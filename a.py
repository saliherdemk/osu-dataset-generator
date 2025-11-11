from Architecture.Dataset import createDataLoader

train_dataloader = createDataLoader("/home/saliherdemk/try_dataset/precomputed/", 4)

for batch_idx, data in enumerate(train_dataloader):
    chunk_audio, diff_rating, hit_obj_data = data
    print(chunk_audio.shape, diff_rating.shape, hit_obj_data.shape)
