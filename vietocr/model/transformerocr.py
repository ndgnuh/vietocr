from vietocr.model.backbone.cnn import CNN
from vietocr.model.seqmodel.transformer import LanguageTransformer
from vietocr.model.seqmodel.seq2seq import Seq2Seq
from vietocr.model.seqmodel.convseq2seq import ConvSeq2Seq
from vietocr.model.stn import SpatialTransformer
from torch import nn


class VietOCR(nn.Module):
    def __init__(self, vocab_size,
                 backbone,
                 cnn_args,
                 transformer_args, seq_modeling='transformer', stn=0):

        super(VietOCR, self).__init__()
        if stn > 0:
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
        else:
            raise('Not Support Seq Model')

    def forward(self, img, tgt_input=None, tgt_key_padding_mask=None, teacher_forcing=False):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        if hasattr(self, "stn"):
            img = self.stn(img)
        src = self.cnn(img)

        if self.seq_modeling == 'transformer':
            outputs = self.transformer(
                src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
        elif self.seq_modeling == 'seq2seq':
            outputs = self.transformer(
                src,
                tgt_input,
                teacher_forcing=teacher_forcing
            )
        elif self.seq_modeling == 'convseq2seq':
            outputs = self.transformer(src, tgt_input)
        return outputs
