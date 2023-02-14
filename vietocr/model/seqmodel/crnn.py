from torch import nn
import torch


class RNNFC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: t b h
        recur, _ = self.rnn(x)
        x = self.fc(recur)
        return x


class CRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        correction: bool = False
    ):
        super().__init__()
        self.correction = correction
        self.rnn_1 = RNNFC(input_size, hidden_size, hidden_size)
        self.rnn_2 = RNNFC(hidden_size, hidden_size, vocab_size)
        if correction:
            self.correction = Correction(vocab_size, hidden_size)

    def forward(self, x):
        x = self.rnn_1(x)
        x = self.rnn_2(x)
        if self.correction:
            x = self.correction(x.argmax(dim=-1))
        x = x.transpose(0, 1)
        return x


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
