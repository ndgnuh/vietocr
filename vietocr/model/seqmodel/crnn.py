from torch import nn


class CRNN(nn.Module):
    def __init__(self, vocab_size, input_size: int, hidden_size: int):
        super().__init__()
        self.rnn1 = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.rnn2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, feature):
        out, _ = self.rnn1(feature)
        out, _ = self.rnn2(out)
        out = self.fc(out)

        # t b h -> b t h
        out = out.transpose(0, 1)
        return out
