import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from TimingModel import TimingModel
from torch.utils.tensorboard import SummaryWriter

from Dataset import createDataLoader


def create_kernel(sigma=4.0, kernel_size=21):
    padding = kernel_size // 2  # match output_size with the input size

    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel = g.unsqueeze(0).unsqueeze(0)

    return kernel, padding


def owbce(
    gt_start_offsets, gt_end_offsets, kernel, kernel_padding
):  # https://arxiv.org/pdf/2403.13254
    gt_start_offsets = gt_start_offsets.reshape(-1, gt_start_offsets.shape[2])
    gt_end_offsets = gt_end_offsets.reshape(-1, gt_end_offsets.shape[2])

    C_s = 1.0
    C_e = 1.0
    alpha_start = (gt_start_offsets != 0).float() * C_s
    alpha_end = (gt_end_offsets != 0).float() * C_e
    alpha = alpha_start + alpha_end

    alpha_conv_input = alpha.unsqueeze(1)

    omega_penalty = F.conv1d(alpha_conv_input, kernel, padding=kernel_padding)

    omega_penalty = omega_penalty.squeeze(1)
    lambda_weight = 1.0

    omega = lambda_weight + omega_penalty
    return omega


def train(dataloader, model, save_to, num_epochs, lr, epoch):
    criterion_bce = nn.BCEWithLogitsLoss(reduction="none")
    kernel, kernel_padding = create_kernel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    if save_to:
        os.makedirs(save_to, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(save_to, "logs"))
    else:
        writer = None

    for epoch in range(epoch, num_epochs):
        epoch_loss = 0
        for data in dataloader:
            gt_hit_logits = data["has_hit"]
            gt_hit_logits = gt_hit_logits.reshape(-1, gt_hit_logits.shape[2])

            optimizer.zero_grad()

            y_pred = model(data["audio"], data["difficulty_rating"])
            y_pred_hit_logits = y_pred[..., 0]
            y_pred_hit_logits = y_pred_hit_logits.reshape(-1, y_pred.shape[2])

            l_bce_per_frame = criterion_bce(y_pred_hit_logits, gt_hit_logits)

            omega = owbce(
                data["start_offsets"], data["end_offsets"], kernel, kernel_padding
            )

            l_owbce_weighted = omega * l_bce_per_frame
            final_owbce_loss = l_owbce_weighted.mean()
            final_owbce_loss.backward()
            optimizer.step()
            epoch_loss += final_owbce_loss.item()

            break
        if save_to:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_to, f"timing_model_{epoch}.pth"))

        avg_loss = epoch_loss / len(dataloader)
        if writer:
            writer.add_scalar("Loss/epoch", avg_loss, epoch)
        print(f"Epoch: {epoch}, Loss: {epoch_loss / len(dataloader)}")
    if writer:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", required=True)
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--num_epochs", default=10)
    parser.add_argument("--load_from", default=None)
    parser.add_argument("--save_to", default=None)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--lr", default=1e-4, type=float)

    args = parser.parse_args()

    dataloader = createDataLoader(args.dataset_folder, int(args.batch_size))

    for data in dataloader:
        chunk_audio, hit_obj_data, diff_rating = data
        print(chunk_audio.shape, hit_obj_data.shape, diff_rating.shape)
        break

    return
    model = TimingModel()
    epoch = 0

    if args.load_from:
        checkpoint = torch.load(args.load_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"] + 1

    if args.mode == "train":
        train(dataloader, model, args.save_to, args.num_epochs, args.lr, epoch)


if __name__ == "__main__":
    main()
