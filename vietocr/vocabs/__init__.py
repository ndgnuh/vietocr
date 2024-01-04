"""The vocab module contains encoding and decoding rules.

The rules contain a specific set of characters. Because of
convenience, the character sets are divided into "languages".
The ideal API would allow an user to specific either:
- a language (basic use case, e.g. English),
- a set of languages (combination of multiple languages or parts, e.g. Vietnamese + Japanese),
- any character sets (no presets satisfy the use case).

The user would also have to specify which types of vocabulary to
use. This package (originally) provides two types of vocabs: the
CTC vocabs and the Seq2Seq vocabs. Each type has its own rule of
encoding.

Due to the refactor, the Seq2Seq vocab is temporary unsupported and
a number of features is not available. Currently, only single language
specification is valid, multiple languages or custom character sets is
a work in progress.
"""
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
    """List of the characters in a supported languages.

    Args:
        lang (str): Name of the presets.

    Returns:
        chars (list[str]): List of characters.
    """
    return vocabs[lang]


def list_languages() -> Set[str]:
    """List supported presets.

    This function has no arguments.

    Returns:
        langs (Set[str]): A set of all the preset names.
    """
    return set(vocabs.keys())


def get_vocab(lang: str, vocab_type: str = "ctc") -> Vocab:
    """Get vocabulary object based on the language and vocab type.

    Args:
        lang (str): The language preset name.
        vocab_type (str): Either "ctc" for CTC vocab or "s2s" for Sequence-to-sequence vocab.

    Returns:
        A :class:`.Vocab` object with the specified type and character set.
    """
    msg = f"Expected vocab types in: {vocab_types.keys()}, got '{vocab_type}'"
    assert vocab_type in vocab_types, msg
    chars = list_characters(lang)
    return vocab_types[vocab_type](chars)
