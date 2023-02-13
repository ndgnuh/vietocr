from torch import nn


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
    def __init__(self, vocab_size, input_size, hidden_size):
        super().__init__()
        self.rnn_1 = RNNFC(input_size, hidden_size, hidden_size)
        self.rnn_2 = RNNFC(hidden_size, hidden_size, vocab_size)

    def forward(self, x):
        x = self.rnn_1(x)
        x = self.rnn_2(x)
        x = x.transpose(0, 1)
        return x
