import os

import torch
import torch.nn as nn
from Decoder import Decoder
from Encoder import Encoder
from tqdm import tqdm


class Seq2Seq:
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4
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

        loss_fn_type = nn.CrossEntropyLoss()
        loss_fn_cont = nn.MSELoss()

        for epoch in range(epochs + 1, num_epochs):
            epoch_loss = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
            for batch_idx, batch in enumerate(pbar):
                audio = batch["audio"].to(device)
                audio_mask = batch["audio_mask"].to(device)
                diff = batch["difficulty_rating"].to(device)
                features = batch["features"].to(device)
                tgt_causal_mask = batch["tgt_causal_mask"].to(device)
                tgt_key_padding_mask = batch["tgt_key_padding_mask"].to(device)

                memory = self.encoder(audio, diff, ~audio_mask)
                type_logits, cont_preds = self.decoder(
                    features,
                    memory,
                    tgt_causal_mask=tgt_causal_mask,
                    tgt_key_padding_mask=~tgt_key_padding_mask,
                    memory_key_padding_mask=~audio_mask,
                )

                type_target = features[..., :3].argmax(dim=-1)
                cont_target = features[..., 3:]

                loss_type = loss_fn_type(type_logits.view(-1, 3), type_target.view(-1))
                loss_cont = loss_fn_cont(cont_preds, cont_target)
                loss = loss_type + loss_cont

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                running_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({"running_loss": f"{running_loss:.4f}"})

            self.scheduler.step()
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
