from typing import Set

from pkg_resources import resource_string

vocabs = {}
vocab_names = ["vietnamese"]

for vocab_name in vocab_names:
    vocab = resource_string("vietocr.vocabs", f"{vocab_name}.txt").decode("utf-8")
    vocabs[vocab_name] = vocab


def list_characters(lang: str) -> str:
    return vocabs[lang]


def list_languages() -> Set[str]:
    return set(vocabs.keys())
