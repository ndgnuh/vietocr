from vietocr.model.backbone.cnn import CNN
from vietocr.model.seqmodel.transformer import LanguageTransformer
from vietocr.model.seqmodel.seq2seq import Seq2Seq
from vietocr.model.seqmodel.convseq2seq import ConvSeq2Seq
from vietocr.model.stn import SpatialTransformer
from torch import nn


class NoSeqDecoderLayer(nn.Module):
    def __init__(self, head_size, num_attention_heads):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(head_size, head_size * 4),
            nn.GELU(),
            nn.Linear(head_size * 4, head_size),
            nn.LayerNorm(head_size)
        )
        self.KV = nn.Linear(head_size, head_size*2)
        self.attn = nn.MultiheadAttention(head_size, num_attention_heads)
        self.norm = nn.LayerNorm(head_size)
        self.act = nn.GELU()

    def forward(self, x, img_feature):
        K, V = self.KV(x).chunk(2, dim=-1)
        x_new, _ = self.attn(img_feature, K, V)
        x = self.norm(x + x_new)
        x = self.fc(x) + x
        return x


class NoSeqDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 head_size,
                 num_attention_heads,
                 num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            NoSeqDecoderLayer(head_size, num_attention_heads)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(head_size, vocab_size)

    def forward(self, feature):
        x = feature
        for layer in self.layers:
            x = layer(x, feature)
        x = self.fc(x)
        x = x.transpose(0, 1)
        return x


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
        elif seq_modeling == 'none' or seq_modeling is None:
            self.transformer = NoSeqDecoder(vocab_size, **transformer_args)
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
        if hasattr(self, "stn"):
            img = self.stn(img)

        # src: [time, batch, hidden]
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
        elif self.seq_modeling == 'none' or self.seq_modeling is None:
            outputs = self.transformer(src)
        return outputs
