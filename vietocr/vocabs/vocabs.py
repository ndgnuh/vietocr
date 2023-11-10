from abc import ABC, abstractmethod
from itertools import groupby
from typing import List

from unidecode import unidecode

DEFAULT_REPACEMENT = {
    "−": "-",  # Dash and hyphen?
    "Ð": "Đ",  # Vietnamese Đ
}


def read_vocab_file(file: str) -> List[str]:
    with open(file) as f:
        vocab = f.read()
        vocab = vocab.strip("\r\n\t")
        return list(vocab)


def replace_at(s: str, i: int, r: str) -> str:
    "Create a new string with one replaced character"
    return f"{s[:i]}{r}{s[i+1:]}"


def sanitize(s: str, valid_chars: List[str], replacement=None) -> str:
    "Remove out-of-vocabularies characters and replace similar characters with one variant"
    if replacement is None:
        replacement = DEFAULT_REPACEMENT

    s = s.replace("\n", " ")
    s = s.replace("\r", " ")
    s = s.replace("  ", " ")

    # Replace until there is no out-of-vocab characeters
    prev = None
    prev_index = 0
    while True:
        prev = s
        for i in range(prev_index, len(s)):
            c = s[i]

            # SKIP WHITE SPACE
            if c in "\n\r":
                continue

            # NORMAL
            if c in valid_chars:
                continue

            # REPLACE
            if c in replacement:
                s = replace_at(s, i, replacement[c])
                prev_index = i
                break

            # PREPLACE WITH UNIDECODE
            if c not in valid_chars:
                replace = unidecode(c)
                # if not in valid_chars, keep it, a blank will be placed there
                if replace in valid_chars:
                    s = replace_at(s, i, replace)
                    prev_index = i
                    break

        if prev == s:
            return s


# def unidecode_string(s, vocab, max_iteration=10):
#     custom_mapping = {"−": "-", "Ð": "Đ"}
#     prev_i = 0
#     n = len(s)
#     prev_s = None
#     # print("==============================")
#     # print("String:", s)
#     while True:
#         # print(s)
#         # If previous string is the same, return
#         if prev_s == s:
#             return s

#         # Loop through and replace
#         all_in_vocab = True
#         for i in range(prev_i, n):
#             c = s[i]
#             # Replace with char inside vocab
#             if c not in vocab:
#                 all_in_vocab = False
#                 # print('- char', c)
#                 replacement = custom_mapping.get(c, unidecode.unidecode(c))
#                 if any([r not in vocab for r in replacement]):
#                     print(
#                         f"Warning: replacement {replacement} not in vocab, using blank")
#                     replacement = ""
#                 prev_s = s
#                 s = replace_at(s, i, replacement)

#                 # Store index to loop efficiently
#                 prev_i = i
#                 continue

#         if all_in_vocab:
#             break
#     # print("Result:", s)
#     return s


class Vocab(ABC):
    @classmethod
    def from_file(cls, file):
        return cls(read_vocab_file(file))

    def __init__(self, chars):
        special_tokens = self.get_special_tokens()
        for i, token in enumerate(special_tokens):
            setattr(self, f"{token}_id", i)

        self.chars = list(chars)
        self.special_tokens = [f"<{token}>" for token in special_tokens]
        self.all_chars = all_chars = list(self.special_tokens) + list(chars)
        self.char2index = {c: i for i, c in enumerate(all_chars)}
        self.index2char = {i: c for i, c in enumerate(all_chars)}
        self.num_special_tokens = len(special_tokens)

    def __len__(self):
        return len(self.all_chars)

    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def batch_encode(self, arr):
        ids = [self.encode(c) for c in arr]
        return ids

    def __str__(self):
        lines = [
            "Chars: " + "".join(self.chars),
            "Special: " + ", ".join(self.special_tokens),
        ]
        return "\n".join(lines)

    def __repr__(self):
        return str(self)

    @abstractmethod
    def get_special_tokens(self):
        ...

    @abstractmethod
    def decode(self, seq):
        ...

    @abstractmethod
    def encode(self, seq, max_length=None):
        ...


class VocabS2S(Vocab):
    def get_special_tokens(self):
        return ["pad", "sos", "eos", "other"]

    def encode(self, chars, max_length=None):
        vocab = self.chars
        chars = sanitize(chars, vocab)
        seq = [self.char2index.get(c, self.other_id) for c in chars]
        if max_length is not None:
            n = len(seq)
            seq = seq[: max_length - 2]
        seq = [self.sos_id] + seq + [self.eos_id]
        if max_length is not None:
            seq = seq + [self.pad_id] * (max_length - 2 - n)
        return seq

    def decode(self, ids):
        last = ids.index(self.eos_id) if self.eos_id in ids else None
        first = ids.index(self.sos_id) if self.sos_id in ids else 0
        if last is not None:
            ids = ids[first:last]
        else:
            ids = ids[first:]
        sent = "".join([self.index2char[i] for i in ids if self.is_normal_id(i)])
        return sent.strip(" ")


class VocabCTC(Vocab):
    def get_special_tokens(self):
        return ["blank"]

    def encode(self, chars):
        chars = sanitize(chars, self.chars)
        ids = [self.blank_id]
        prev = None
        for c in chars:
            if prev == c:
                ids.append(self.blank_id)
            ids.append(self.char2index.get(c, self.blank_id))
            prev = c
        return ids

    def decode(self, ids: List[int]) -> str:
        "perform Greedy CTC decoding"

        collapsed = [self.index2char[i] for i, _ in groupby(ids) if i != self.blank_id]
        return "".join(collapsed).strip(" ")
