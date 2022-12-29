from os import path, listdir
from .vocab import Vocab, VocabS2S

thisdir = path.dirname(__file__)
vocab_files = {
    path.splitext(f)[0]: path.join(thisdir, f)
    for f in listdir(thisdir) if f.endswith(".vocab")
}


def read_vocab_file(file):
    with open(file) as f:
        vocab = f.read()
        vocab = vocab.strip("\r\n\t")
        return list(vocab)


def get_chars(lang):
    if lang not in vocab_files:
        raise ValueError(
            f"Bad language {lang}, availabe languages are {list(vocab_files.keys())}")

    return read_vocab_file(vocab_files[lang])


def get_vocab(lang):
    return VocabS2S(get_chars(lang))
