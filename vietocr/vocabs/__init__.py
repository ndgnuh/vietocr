from typing import Set

from pkg_resources import resource_string

from .vocabs import Vocab, VocabCTC, VocabS2S

vocab_types = dict(ctc=VocabCTC, seq2seq=VocabS2S)
vocabs = {}
vocab_names = ["vietnamese"]

for vocab_name in vocab_names:
    vocab = resource_string("vietocr.vocabs", f"{vocab_name}.txt").decode("utf-8")
    vocabs[vocab_name] = vocab


def list_characters(lang: str) -> str:
    return vocabs[lang]


def list_languages() -> Set[str]:
    return set(vocabs.keys())


def get_vocab(lang: str, vocab_type: str = 'ctc'):
    msg = f"Expected vocab types in: {vocab_types.keys()}, got '{vocab_type}'"
    assert vocab_type in vocab_types, msg
    chars = list_characters(lang)
    return vocab_types[vocab_type](chars)
