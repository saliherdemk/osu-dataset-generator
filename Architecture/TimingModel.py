import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import N_MELS


class AudioEncoder(nn.Module):
    def __init__(self, out_channels=32, kernel_size=3):
        super().__init__()
        self.c1 = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.b1 = nn.BatchNorm2d(out_channels)

        self.c2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=kernel_size,
            padding="same",
        )
        self.b2 = nn.BatchNorm2d(out_channels * 2)

        self.c3 = nn.Conv2d(
            in_channels=out_channels * 2,
            out_channels=out_channels * 4,
            kernel_size=kernel_size,
            padding="same",
        )
        self.b3 = nn.BatchNorm2d(out_channels * 4)

        self.maxPooling = nn.MaxPool2d(kernel_size=(1, 2))

        self.output_size = int(out_channels * 4 * N_MELS / 4)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = torch.relu(x)

        x = self.maxPooling(x)

        x = self.c2(x)
        x = self.b2(x)
        x = torch.relu(x)

        x = self.maxPooling(x)

        x = self.c3(x)
        x = self.b3(x)
        x = torch.relu(x)

        x = x.permute(0, 2, 1, 3)
        x = x.flatten(start_dim=2)

        return x


class Decoder(nn.Module):
    def __init__(self, input_size=2048 + 64, rnn_hidden_size=256, rnn_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.output_size = rnn_hidden_size * 2

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return gru_out


class TimingModel(nn.Module):
    def __init__(self, dr_embedding_size=64):
        super().__init__()
        self.encoder = AudioEncoder()
        self.decoder = Decoder(int(self.encoder.output_size))
        self.dropout = nn.Dropout(0.1)

        self.dr_embed = nn.Sequential(nn.Linear(1, dr_embedding_size), nn.ReLU())

        self.gamma_head = nn.Linear(dr_embedding_size, self.encoder.output_size)
        self.beta_head = nn.Linear(dr_embedding_size, self.encoder.output_size)

        self.classification_head = nn.Linear(self.decoder.output_size, 7)

    def forward(self, x, diff_rating):

        dr_x = self.dr_embed(diff_rating)
        # dr_x = dr_x.unsqueeze(1)
        # dr_x = dr_x.repeat(1, x.size(1), 1)
        # x = torch.cat((x, dr_x), dim=2)

        gamma = self.gamma_head(dr_x).unsqueeze(1)
        beta = self.beta_head(dr_x).unsqueeze(1)

        audio_features = self.encoder(x)
        x = (audio_features * gamma) + beta

        # x = self.dropout(x)
        x = self.decoder(x)

        final_output = self.classification_head(x)

        return final_output

    def predict(self, chunks, diff_rating):
        diff_rating = torch.tensor([diff_rating], dtype=torch.float32)
        diff_rating = torch.full(
            (chunks.shape[0], 1), diff_rating.item(), dtype=torch.float32
        )
        with torch.no_grad():
            output = self.forward(chunks, diff_rating)
            probs = torch.sigmoid(output)

        return probs
