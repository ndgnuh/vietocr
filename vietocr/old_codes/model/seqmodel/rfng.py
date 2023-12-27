# Experimental no recurrent decoder
from torch import nn


class RefineAndGuessLayer(nn.Module):
    def __init__(self, vocab_size, head_size, num_attention_heads: int):
        super().__init__()
        self.refine = nn.MultiheadAttention(head_size, num_attention_heads)
        self.guess = nn.Linear(head_size, vocab_size)

    def forward(self, feature_maps, embeds, hiddens):
        refined, _ = self.refine(feature_maps, embeds, hiddens)
        guess = self.guess(refined)
        return refined, guess


class RefineAndGuess(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        head_size: int,
        num_layers: int = 3,
        num_attention_heads: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, head_size)
        self.first_try = nn.Linear(head_size, vocab_size)
        self.guesses = nn.ModuleList([
            RefineAndGuessLayer(vocab_size, head_size, num_attention_heads)
            for _ in range(num_layers)
        ])

    def forward(self, feature_maps):
        # feature_maps: t, b, h
        first_try = self.first_try(feature_maps).argmax(dim=-1)
        hiddens = feature_maps
        embeds = self.embedding(first_try)
        last_try = None
        for guess in self.guesses:
            hiddens, last_try = guess(feature_maps, embeds, hiddens)
            embeds = self.embedding(last_try.argmax(dim=-1))

        # t, b, h -> b, t, h
        last_try = last_try.transpose(0, 1)
        return last_try
