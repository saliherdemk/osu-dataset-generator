import argparse
import os

import torch
from kornia.losses import binary_focal_loss_with_logits
from TimingModel import TimingModel
from tqdm import tqdm

from Architecture.Dataset import (
    createDataLoader,
    pred_to_df,
    prepare_audio_for_prediction,
)


def train(dataloader, model, optimizer, save_to, num_epochs, start_epoch, device):
    kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": "mean"}

    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        epoch_loss = 0.0

        for _batch_idx, data in enumerate(dataloader):
            chunk_audio, diff_rating, hit_obj_data = data

            chunk_audio = chunk_audio.to(device)
            diff_rating = diff_rating.to(device)
            hit_obj_data = hit_obj_data.to(device)

            optimizer.zero_grad()

            res = model(chunk_audio, diff_rating)

            has_hit_pred = res
            has_hit_gt = hit_obj_data

            loss = binary_focal_loss_with_logits(
                has_hit_pred.reshape(-1, has_hit_pred.shape[2]),
                has_hit_gt.reshape(-1, has_hit_gt.shape[2]),
                **kwargs,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}/{num_epochs} | Avg Loss: {avg_loss:.6f}")
        if save_to and epoch % 1 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                },
                os.path.join(save_to, f"epoch_{epoch}.pt"),
            )


def predict_audio(model, audio_path, save_to, diff_rating, device):
    model.eval()
    chunks = prepare_audio_for_prediction(audio_path)
    chunks = chunks.unsqueeze(1).to(device)
    probs = model.predict(chunks, diff_rating)
    df = pred_to_df(probs)

    filename = os.path.basename(audio_path)
    name = os.path.splitext(filename)[0]

    df.to_csv(os.path.join(save_to, name + ".csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", default=None)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--load_from", default=None)
    parser.add_argument("--save_to", default=None)
    parser.add_argument("--mode", default="train", choices=["train", "predict"])
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--audio_path", default=None)
    parser.add_argument("--diff_rating", default=None, type=float)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TimingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0

    if args.load_from:
        checkpoint = torch.load(args.load_from, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])

    if args.mode == "train":
        if args.dataset_folder is None:
            raise ValueError("--dataset_folder is required for training mode")
        if start_epoch > args.num_epochs:
            raise ValueError(
                f"The model has already been trained for {start_epoch} epochs. Use --num_epochs with a higher value to train further."
            )

        dataloader = createDataLoader(args.dataset_folder, args.batch_size)
        train(
            dataloader,
            model,
            optimizer,
            args.save_to,
            args.num_epochs,
            start_epoch,
            device,
        )
    else:
        if args.audio_path is None:
            raise ValueError("--audio_path is required for prediction mode")
        if args.diff_rating is None:
            raise ValueError("--diff_rating is required for prediction mode")
        if args.save_to is None:
            raise ValueError("--save_to is required for prediction mode")

        predict_audio(model, args.audio_path, args.save_to, args.diff_rating, device)


if __name__ == "__main__":
    main()
