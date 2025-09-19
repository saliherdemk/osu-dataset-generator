import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=768,
        emb_dim=1023,
        max_seq_len=1000,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=emb_dim, kernel_size=1)

        d_model = emb_dim + 1

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model)
        )  # [1, seq_len, d_model]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, diff_rating, audio_mask):
        # x: [batch, seq_len, d_model]
        x = x.permute(0, 2, 1)  # [batch, d_model, seq_len]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)  # [batch, d_model, seq_len]
        x = x.permute(0, 2, 1)  # [batch, seq_len, d_model]

        seq_len = x.size(1)
        diff_rating = diff_rating.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x, diff_rating], dim=-1)

        x = x + self.pos_embedding[:, :seq_len, :]
        out = self.encoder(x, src_key_padding_mask=audio_mask)

        return out
