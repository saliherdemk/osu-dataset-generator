import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 2  # mel + diff rating
        self.c1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))

        self.c2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

    def forward(self, audio, diff_rating):
        batch_size, num_chunks, time, mel = audio.shape

        x = torch.stack([audio, diff_rating], dim=1)

        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size * num_chunks, 2, time, mel)

        x = torch.relu(self.c1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        x = torch.relu(self.c2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, num_chunks, x.size(1), -1)

        return x


class TemporalEncoder(nn.Module):
    def __init__(
        self, input_size=1024, output_size=3, hidden_size=512, num_layers=2, dropout=0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout,
        )
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        chunk_num = x.shape[1]
        outputs = []
        h, c = None, None

        for i in range(chunk_num):
            chunk = x[:, i, :, :]
            chunk = chunk.permute(1, 0, 2)
            out, (h, c) = self.lstm(chunk, (h, c) if h is not None else None)

            out = out.permute(1, 0, 2)
            out = torch.sigmoid(self.fc(out))
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        return outputs


class TimingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.audioEncoder = AudioEncoder()
        self.temporalEncoder = TemporalEncoder()

    def forward(self, audio, diff_rating):
        x = self.audioEncoder(audio, diff_rating)
        x = self.temporalEncoder(x)
        return x
