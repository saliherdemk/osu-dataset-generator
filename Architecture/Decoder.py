import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        d_model=1024,
        num_types=3,
        max_seq_len=1000,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(5, d_model)  # 3 type + 2 timing

        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.type_head = nn.Linear(d_model, num_types)
        self.cont_head = nn.Linear(d_model, 2)

    def forward(
        self,
        tgt,
        memory,
        tgt_causal_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
    ):
        x = self.input_proj(tgt.float())

        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        type_logits = self.type_head(out)
        cont_preds = self.cont_head(out)

        return type_logits, cont_preds
