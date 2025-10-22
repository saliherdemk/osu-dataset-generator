import argparse
import os

import torch
from kornia.losses import binary_focal_loss_with_logits
from TimingModel import TimingModel
from tqdm import tqdm

from Architecture.Dataset import createDataLoader


def train(dataloader, model, save_to, load_from, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": "mean"}

    start_epoch = 0
    if load_from:
        checkpoint = torch.load(load_from, map_location=torch.device("cpu"))
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])

    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        epoch_loss = 0.0

        for batch_idx, data in enumerate(dataloader):
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
            # print(epoch_loss)

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.6f}")
        if save_to and epoch % 1 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                },
                os.path.join(save_to, f"epoch_{epoch}.pt"),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", required=True)
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--load_from", default=None)
    parser.add_argument("--save_to", default=None)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--lr", default=1e-4, type=float)

    args = parser.parse_args()
    dataloader = createDataLoader(args.dataset_folder, int(args.batch_size))

    model = TimingModel()

    train(dataloader, model, args.save_to, args.load_from, args.num_epochs, args.lr)


if __name__ == "__main__":
    main()
