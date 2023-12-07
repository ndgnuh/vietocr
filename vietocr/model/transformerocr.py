import torch
from torch import nn
from torch.nn import functional as F

from vietocr.model.backbone.cnn import CNN
from vietocr.model.seqmodel.c3rnn import C3RNN
from vietocr.model.seqmodel.convseq2seq import ConvSeq2Seq
from vietocr.model.seqmodel.seq2seq import Seq2Seq
from vietocr.model.seqmodel.transformer import LanguageTransformer
from vietocr.model.stn import SpatialTransformer

from .seqmodel.crnn import CRNN


class NoneDecoder(nn.Module):
    def __init__(self, vocab_size, *a, **k):
        super().__init__()
        self.vocab_size = vocab_size
        self.err_msg = f"Output doesn't match vocab size of {self.vocab_size}"

    def forward(self, x, *a, **k):
        x = x.transpose(1, 0)
        assert x.shape[-1] == self.vocab_size, self.err_msg
        return x


decoder_map = dict(
    transformer=LanguageTransformer,
    seq2seq=Seq2Seq,
    convseq2seq=ConvSeq2Seq,
    crnn=CRNN,
    none=NoneDecoder,
)
decoder_map[None] = NoneDecoder


class VietOCR(nn.Module):
    def __init__(
        self,
        vocab_size,
        backbone,
        cnn_args,
        transformer_args,
        seq_modeling="transformer",
        stn=None,
    ):
        super(VietOCR, self).__init__()
        self.vocab_size = vocab_size
        self.stn = SpatialTransformer(stn)
        self.cnn = CNN(backbone, **cnn_args)
        self.seq_modeling = seq_modeling

        DecoderLayer = decoder_map[seq_modeling]
        self.transformer = DecoderLayer(vocab_size, **transformer_args)

    def forward(self, img, tgt=None, tgt_key_padding_mask=None, teacher_forcing=False):
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

        outputs = self.transformer(
            src, tgt=tgt, tgt_key_padding_mask=tgt_key_padding_mask
        )
        return outputs


class OnnxWrapper(nn.Module):
    def __init__(self, model, image_height):
        super().__init__()
        self.model = model
        self.image_height = image_height
        self.eval()

    def example_input(self):
        return torch.rand(1, 3, self.image_height, 512)

    def forward(self, image):
        outputs = self.model(image)
        outputs = torch.softmax(outputs, dim=-1)
        scores, classes = torch.max(outputs, dim=-1)
        return classes, scores

    def export(self, output_file: str):
        torch.onnx.export(
            self.eval(),
            self.example_input(),
            output_file,
            do_constant_folding=True,
            input_names=["images"],
            dynamic_axes=dict(images=[0, 3]),
        )
