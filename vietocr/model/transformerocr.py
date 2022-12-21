from vietocr.model.backbone.cnn import CNN
from vietocr.model.seqmodel.transformer import LanguageTransformer
from vietocr.model.seqmodel.seq2seq import Seq2Seq
from vietocr.model.seqmodel.convseq2seq import ConvSeq2Seq
from torch import nn


class VietOCR(nn.Module):
    def __init__(self, vocab_size,
                 backbone,
                 cnn_args,
                 transformer_args, seq_modeling='transformer'):

        super(VietOCR, self).__init__()

        self.backbone = CNN(backbone, **cnn_args)
        self.seq_modeling = seq_modeling

        if seq_modeling == 'transformer':
            self.head = LanguageTransformer(
                vocab_size, **transformer_args)
        elif seq_modeling == 'seq2seq':
            self.head = Seq2Seq(vocab_size, **transformer_args)
        elif seq_modeling == 'convseq2seq':
            self.head = ConvSeq2Seq(vocab_size, **transformer_args)
        else:
            raise('Not Support Seq Model')

    def forward(self, image, target=None, target_mask=None):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        source = self.backbone(image)

        #
        target = target.transpose(0, 1)

        if self.seq_modeling == 'transformer':
            outputs = self.head(
                source,
                target,
                tgt_key_padding_mask=1 - target_mask
            )
        elif self.seq_modeling == 'seq2seq':
            outputs = self.head(source, target)
        elif self.seq_modeling == 'convseq2seq':
            outputs = self.head(source, target)
        return outputs
