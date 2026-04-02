import argparse
import os

import numpy as np
import torch
from kornia.losses import binary_focal_loss_with_logits
from sklearn.metrics import f1_score
from TimingModel import TimingModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Architecture.Dataset import (
    createDataLoader,
    pred_to_df,
    prepare_audio_for_prediction,
)


def save_model(epoch, model, optimizer, scheduler, save_to):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        os.path.join(save_to, f"epoch_{epoch}.pt"),
    )


def train(
    train_dataloader,
    eval_dataloader,
    model,
    optimizer,
    scheduler,
    save_to,
    num_epochs,
    start_epoch,
    log_dir,
    device,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": "mean"}

    start_epoch = 0

    writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        train_epoch_loss = 0.0

        for batch_idx, data in enumerate(train_dataloader):
            chunk_audio, diff_rating, hit_obj_data = data

            chunk_audio = chunk_audio.to(device)
            diff_rating = diff_rating.to(device)
            hit_obj_data = hit_obj_data.to(device)

            res = model(chunk_audio, diff_rating)
            loss = binary_focal_loss_with_logits(
                res.reshape(-1, res.shape[2]),
                hit_obj_data.reshape(-1, hit_obj_data.shape[2]),
                **kwargs,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_train_loss:.6f}")
        if writer:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)

        if epoch % 200 == 0:
            save_model(epoch, model, optimizer, scheduler, save_to)

        if not eval_dataloader:
            continue

        model.eval()
        with torch.no_grad():
            val_epoch_loss = 0.0
            all_preds = []
            all_targets = []
            for batch_idx, data in enumerate(eval_dataloader):
                chunk_audio, diff_rating, hit_obj_data = data

                chunk_audio = chunk_audio.to(device)
                diff_rating = diff_rating.to(device)
                hit_obj_data = hit_obj_data.to(device)

                res = model(chunk_audio, diff_rating)
                loss = binary_focal_loss_with_logits(
                    res.reshape(-1, res.shape[2]),
                    hit_obj_data.reshape(-1, hit_obj_data.shape[2]),
                    **kwargs,
                )

                val_epoch_loss += loss.item()

                probs = torch.sigmoid(res)
                preds = (probs > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(hit_obj_data.cpu().numpy())

            avg_val_loss = val_epoch_loss / len(eval_dataloader)
            scheduler.step(avg_val_loss)
            if writer:
                writer.add_scalar("Loss/val", avg_val_loss, epoch)

            all_preds = np.concatenate(all_preds).ravel()
            all_targets = np.concatenate(all_targets).ravel()
            f1 = f1_score(all_targets, all_preds)
            if writer:
                writer.add_scalar("F1/val", f1, epoch)


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
    parser.add_argument("--train_dataset_folder", default=None)
    parser.add_argument("--eval_dataset_folder", default=None)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--load_from", default=None)
    parser.add_argument("--save_to", default=None)
    parser.add_argument("--mode", default="train", choices=["train", "predict"])
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--audio_path", default=None)
    parser.add_argument("--diff_rating", default=None, type=float)
    parser.add_argument("--log_dir", default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TimingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    start_epoch = 0

    if args.load_from:
        checkpoint = torch.load(args.load_from, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    if args.mode == "train":
        if args.train_dataset_folder is None:
            raise ValueError("--train_dataset_folder is required for training mode")
        if args.save_to is None:
            raise ValueError("--save_to is required for training mode")
        if start_epoch > args.num_epochs:
            raise ValueError(
                f"The model has already been trained for {start_epoch} epochs. Use --num_epochs with a higher value to train further."
            )
        bs = int(args.batch_size)

        train_dataloader = createDataLoader(args.train_dataset_folder, bs)
        eval_dataloader = (
            createDataLoader(args.eval_dataset_folder, bs)
            if args.eval_dataset_folder
            else None
        )

        train(
            train_dataloader,
            eval_dataloader,
            model,
            optimizer,
            scheduler,
            args.save_to,
            args.num_epochs,
            start_epoch,
            args.log_dir,
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
