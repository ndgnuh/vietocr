from torch import nn


class LinearHead(nn.Linear):
    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = True):
        super().__init__(hidden_size, vocab_size, bias)


MODULES = {"linear": LinearHead, "fc": LinearHead}
