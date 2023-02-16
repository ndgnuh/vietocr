from torch import nn
from torch.nn import functional as F
import torch


class RNNFC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: t b h
        recur, _ = self.rnn(x)
        recur = self.tanh(recur)
        x = self.fc(recur)
        return x


class CRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        correction: bool = False,
        reparam: bool = False
    ):
        super().__init__()
        self.correction = correction
        self.rnn_1 = RNNFC(input_size, hidden_size, hidden_size)
        self.rnn_2 = RNNFC(hidden_size, hidden_size, vocab_size)
        if correction:
            self.correction = Correction(vocab_size, hidden_size)

        if reparam:
            self.reparam = Reparam(hidden_size)
        else:
            self.reparam = nn.Identity()

    def forward(self, x):
        x = self.rnn_1(x)
        x = self.reparam(x)
        x = self.rnn_2(x)
        if self.correction:
            x = self.correction(x.argmax(dim=-1))
        x = x.transpose(0, 1)
        return x


class Reparam(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mu = nn.Linear(hidden_size, hidden_size)
        self.sigma = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        mu = self.mu(x)
        sigma = self.sigma(x)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return (mu + eps*std)


class Correction(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.tanh = nn.Tanh()
        self.enc_embed = nn.Embedding(vocab_size, hidden_size)
        self.enc_rnn = nn.LSTM(hidden_size, hidden_size)
        self.enc_mu = nn.Linear(hidden_size, hidden_size)
        self.enc_sigma = nn.Linear(hidden_size, hidden_size)
        self.dec_rnn = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
        self.dec_fc = nn.Linear(hidden_size * 2, vocab_size)

    def encode(self, x):
        x = self.enc_embed(x)
        output, hidden = self.enc_rnn(x)
        output = self.tanh(output)
        mu = self.enc_mu(output)
        sigma = self.enc_sigma(output)
        return mu, sigma

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return (mu + eps*std)

    def decode(self, x):
        output, _ = self.dec_rnn(x)
        output = self.tanh(output)
        output = self.dec_fc(output)
        return output

    def forward(self, chars):
        mu, sigma = self.encode(chars)
        z = self.reparameterize(mu, sigma)
        out = self.decode(z)
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x, memories = self.rnn(x)
        x = self.fc(x)
        # 2, b, h -> 1, h, b * 2
        memories = torch.cat([memories[0], memories[1]], dim=-1)
        return x, memories


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Linear(vocab_size, hidden_size)
        self.attn_weight = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.attn_combine = nn.Linear(hidden_size * 3, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, memories):
        x = self.embed(x)

        # 1, b, 2h -> t b 2h
        memories = memories.repeat([x.shape[0], 1, 1])

        # output: t, b, h
        attn_weights = self.attn_weight(torch.cat([x, memories], dim=-1))
        attn_applied = attn_weights * memories
        output = torch.cat([x, attn_applied], dim=-1)
        output = self.attn_combine(output)
        output = F.relu(output)

        # t, b, 2h -> 1, b, 2h -> 2 b h
        hidden = attn_applied.mean(dim=0)
        hidden = torch.stack(hidden.chunk(2, dim=-1), dim=0)
        output, _ = self.rnn(output, hidden)
        output = self.out(output)
        return output
#         x = self.embed(x.argmax(dim=-1))


class AttnCRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_size)
        self.decoder = Decoder(vocab_size, hidden_size)

    def forward(self, x):
        x, memories = self.encoder(x)
        x = self.decoder(x, memories)
        x = x.transpose(0, 1)
        return x
