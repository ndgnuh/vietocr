from vietocr.model.backbone.cnn import CNN
from vietocr.model.seqmodel.transformer import LanguageTransformer
from vietocr.model.seqmodel.seq2seq import Seq2Seq
from vietocr.model.seqmodel.convseq2seq import ConvSeq2Seq
from vietocr.model.seqmodel.rfng import RefineAndGuess
from vietocr.model.stn import SpatialTransformer
from .seqmodel.crnn import CRNN, AttnCRNN
from torch import nn
from torch.nn import functional as F


class FC(nn.Module):
    def __init__(self, vocab_size, head_size):
        super().__init__()
        self.fc = nn.Linear(head_size, vocab_size)

    def forward(self, x):
        x = self.fc(x)
        # t b h -> b t h
        x = x.transpose(0, 1)
        return x


class VietOCR(nn.Module):
    def __init__(self, vocab_size,
                 backbone,
                 cnn_args,
                 transformer_args,
                 seq_modeling='transformer',
                 stn=None):

        super(VietOCR, self).__init__()
        self.stn = SpatialTransformer(stn)
        self.cnn = CNN(backbone, **cnn_args)
        self.seq_modeling = seq_modeling

        if seq_modeling == 'transformer':
            self.transformer = LanguageTransformer(
                vocab_size, **transformer_args)
        elif seq_modeling == 'seq2seq':
            self.transformer = Seq2Seq(vocab_size, **transformer_args)
        elif seq_modeling == 'convseq2seq':
            self.transformer = ConvSeq2Seq(vocab_size, **transformer_args)
        elif seq_modeling == 'rfng':
            self.transformer = RefineAndGuess(vocab_size, **transformer_args)
        elif seq_modeling == 'fcrnn':
            self.transformer = FCRNN(vocab_size, **transformer_args)
        elif seq_modeling == 'crnn':
            self.transformer = CRNN(vocab_size, **transformer_args)
        elif seq_modeling == 'atn-crnn':
            self.transformer = AttnCRNN(vocab_size, **transformer_args)
        elif seq_modeling == 'none' or seq_modeling is None:
            self.transformer = nn.Identity()
        else:
            raise('Not Support Seq Model')

    def forward(
        self,
        img,
        tgt_input=None,
        tgt_key_padding_mask=None,
        teacher_forcing=False
    ):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        img = self.stn(img)

        # src: [time, batch, hidden]
        src = self.cnn(img)

        if self.seq_modeling == 'transformer':
            outputs = self.transformer(
                src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
        elif self.seq_modeling.endswith('crnn'):
            outputs = self.transformer(src)
        elif self.seq_modeling == 'seq2seq':
            outputs = self.transformer(
                src,
                tgt_input,
                teacher_forcing=teacher_forcing
            )
        elif self.seq_modeling == 'convseq2seq':
            outputs = self.transformer(src, tgt_input)
        elif self.seq_modeling == 'rfng':
            outputs = self.transformer(src)
        elif self.seq_modeling == 'fcrnn':
            outputs = self.transformer(src)
        elif self.seq_modeling == 'none' or self.seq_modeling is None:
            outputs = self.transformer(src)
        return outputs
