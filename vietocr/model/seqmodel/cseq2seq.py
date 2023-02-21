from torch import nn
import torch
import random

TF = (True, False)


def flip():
    return random.choice(TF)


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input_ids):
        embed = self.embedding(input_ids)

        # t b h, t b vocab_size
        output, hidden = self.gru(embed)

        # 2 b h -> 1 b 2h
        hidden = torch.cat(hidden.chunk(2, dim=0), dim=-1)
        hidden = self.tanh(self.fc(hidden))
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(hidden_size * 3, hidden_size)
        self.fc = nn.Linear(hidden_size * 4, vocab_size)

        # Luong Attention
        self.attn_weights = nn.Linear(hidden_size * 3, hidden_size)
        self.attn_value = nn.Linear(hidden_size, 1, bias=False)
        self.attn_energy = nn.Tanh()
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, input, hidden, encoder_outputs):
        # embed: t b h
        embed = self.embedding(input)
        embed = self.dropout(embed)

        # attention weights: b 1 t
        attn_weights = torch.cat([
            hidden.repeat([encoder_outputs.size(0), 1, 1]),
            encoder_outputs],
            dim=-1)
        attn_weights = self.attn_weights(attn_weights)
        attn_weights = self.attn_energy(attn_weights)
        attn_weights = self.attn_value(attn_weights)
        attn_weights = self.attn_softmax(attn_weights)
        # t b 1 -> b 1 t
        attn_weights = attn_weights.permute(1, 2, 0)

        # Context vector
        # b 1 t * b t h -> b 1 h
        ctx = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))
        # b 1 h -> 1 b h
        ctx = ctx.permute(1, 0, 2)

        # output: 1 b 3h
        output = torch.cat([embed, ctx], dim=2)
        output, hidden = self.rnn(output)
        # output: 1 b 4h
        output = torch.cat([output, ctx, embed], dim=2)
        output = self.fc(output)
        return output


class ReZeroCorrectionSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.enc = EncoderRNN(vocab_size, hidden_size)
        self.dec = DecoderRNN(vocab_size, hidden_size)
        self.alpha = nn.Parameter(torch.tensor(-1e12))

    def forward(self, x):
        # Train on the original output with 50%
        # ensure that it is somewhat legit OCR outputs too
        if self.training and flip():
            x = x.transpose(0, 1)
            return x

        input_ids = x.argmax(dim=-1)
        enc_outputs, hidden = self.enc(input_ids)
        outputs = []
        for x_t in x:
            input = x_t.argmax(dim=-1).unsqueeze(0)
            output = self.dec(input, hidden, enc_outputs)
            alpha = torch.sigmoid(self.alpha)
            output = output * alpha + x_t * (1 - alpha)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.transpose(0, 1)
        return outputs
