# This modules implement wrapper for existing losses
# Loss wrappers should takes batch first inputs
# The purpose is to:
# - provide some defaults over the nn.Losses
# - provide uniform interface for the trainer
# All the losses now have first format

import torch
from torch import nn


class CTCLoss(nn.Module):
    # Reference:
    # https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss

    def __init__(self, *a, **k):
        super().__init__()
        k.setdefault("zero_infinity", True)
        self.ctc = nn.CTCLoss(*a, **k)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, outputs, targets, target_lengths):
        # outputs: [batch, time, class]
        # targets: [batch, max_length]
        # target lengths: [batch]

        # To [time, batch, class]
        outputs = outputs.transpose(0, 1)

        # Log softmax as nn.CTCLoss requires
        logits = self.log_softmax(outputs)

        # target_lengths: [batch]
        # target_lengths = torch.count_nonzero(targets != self.ctc.blank, dim=1)

        # input_lengths: [batch]
        # use time * batch for now
        input_lengths = torch.tensor(
            [logits.shape[0]] * logits.shape[1], device=logits.device
        )

        # ctc loss
        loss = self.ctc(logits, targets, input_lengths, target_lengths)
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        k.setdefault("label_smoothing", 0.1)
        k.setdefault("ignore_index", 0)
        self.loss = nn.CrossEntropyLoss(*a, **k)

    def forward(self, outputs, targets, target_lengths):
        # outputs: [batch, time, num_class]
        # targets: [batch, time]
        # target_lengths: [batch], not that it's matter though

        # to: [num_class, batch, time]
        outputs = outputs.transpose(-1, 1)
        return self.loss(outputs, targets)


MODULES = {"cross_entropy": CrossEntropyLoss, "ctc": CTCLoss}


def get_loss_function(name, options, vocab):
    return eval(name)(vocab, **options)
