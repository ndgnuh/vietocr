from torch import nn


class CRNN(nn.Module):
    def __init__(self, vocab_size, input_size: int, hidden_size: int,
                 bidirectional: bool = True, rnn_type: str = 'GRU'):
        super().__init__()
        RnnLayer = getattr(nn, rnn_type)
        d = 2 if bidirectional else 1
        self.rnn1 = RnnLayer(input_size, hidden_size,
                             bidirectional=bidirectional)
        self.rnn2 = RnnLayer(hidden_size * d, hidden_size,
                             bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * d, vocab_size)

    def forward(self, feature, *a, **k):
        out, _ = self.rnn1(feature)
        out, _ = self.rnn2(out)
        out = self.fc(out)

        # t b h -> b t h
        out = out.transpose(0, 1)
        return out
