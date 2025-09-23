import os

import torch
import torch.nn as nn
from Decoder import Decoder
from Encoder import Encoder
from tqdm import tqdm


class Seq2Seq:
    def __init__(self, lr):
        dropout = 0.0
        n_head = 4
        num_layers = 4
        self.encoder = Encoder(dropout=dropout, nhead=n_head, num_layers=num_layers)
        self.decoder = Decoder(dropout=dropout, nhead=n_head, num_layers=num_layers)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["epoch"]

    def train(self, dataloader, checkpoint_dir=None, num_epochs=10, epochs=0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(device)
        self.decoder.to(device)

        self.encoder.train()
        self.decoder.train()

        loss_fn_cont = nn.MSELoss()

        for epoch in range(epochs + 1, num_epochs):

            epoch_loss, epoch_type_loss, epoch_cont_loss = 0, 0, 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
            for batch_idx, batch in enumerate(pbar):
                audio = batch["audio"].to(device)
                audio_mask = batch["audio_mask"].to(device)
                diff = batch["difficulty_rating"].to(device)
                features = batch["features"].to(device)
                tgt_causal_mask = batch["tgt_causal_mask"].to(device)
                tgt_key_padding_mask = batch["tgt_key_padding_mask"].to(device)

                memory = self.encoder(audio, diff, ~audio_mask)
                preds = self.decoder(
                    features,
                    memory,
                    tgt_causal_mask=tgt_causal_mask,
                    tgt_key_padding_mask=~tgt_key_padding_mask,
                    memory_key_padding_mask=~audio_mask,
                )

                cont_target = features

                loss = loss_fn_cont(preds, cont_target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                running_loss = epoch_loss / (batch_idx + 1)
                running_type = epoch_type_loss / (batch_idx + 1)
                running_cont = epoch_cont_loss / (batch_idx + 1)

                pbar.set_postfix(
                    {
                        "loss": f"{running_loss:.4f}",
                        "type": f"{running_type:.4f}",
                        "cont": f"{running_cont:.4f}",
                    }
                )
            # self.scheduler.step()
            print(f"Epoch {epoch} avg loss: {epoch_loss / len(dataloader):.4f}")
            if checkpoint_dir:
                torch.save(
                    {
                        "encoder_state_dict": self.encoder.state_dict(),
                        "decoder_state_dict": self.decoder.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
                )

    @torch.no_grad()
    def generate_beatmap(model, audio, diff_rating, max_len=1000):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.encoder.eval()
        model.decoder.eval()
        audio = audio.to(device)
        diff_rating = diff_rating.to(device)

        audio_mask = torch.ones(audio.shape[:2], dtype=torch.bool, device=device)
        memory = model.encoder(audio, diff_rating, ~audio_mask)

        tgt_seq = torch.zeros(1, 1, 3, device=device)

        generated = []

        for t in range(max_len):
            seq_len = tgt_seq.size(1)

            tgt_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1
            )
            tgt_key_padding_mask = torch.zeros(
                1, seq_len, dtype=torch.bool, device=device
            )
            memory_key_padding_mask = torch.zeros(
                1, memory.size(1), dtype=torch.bool, device=device
            )

            out = model.decoder(
                tgt=tgt_seq,
                memory=memory,
                tgt_causal_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            next_step = out[:, -1, :]

            next_type = next_step[:, 0].round().clamp(0, 2)
            next_start_end = next_step[:, 1:]  # keep floats

            next_tgt = torch.cat([next_type.unsqueeze(-1), next_start_end], dim=-1)
            generated.append(next_tgt.squeeze(0).cpu())

            tgt_seq = torch.cat([tgt_seq, next_tgt.unsqueeze(1)], dim=1)

        generated_seq = torch.stack(generated, dim=0)
        return generated_seq
