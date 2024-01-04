"""Vocabulary coders implementation."""
from abc import ABC, abstractmethod
from itertools import groupby
from typing import List, Optional

from unidecode import unidecode

DEFAULT_REPACEMENT = {
    "−": "-",  # Dash and hyphen?
    "Ð": "Đ",  # Vietnamese Đ
}
"""This is the replacement for characters that have different code points but look alike."""


def read_vocab_file(file: str) -> List[str]:
    """Read vocabulary's characters from a file.

    This function is not used any more.

    Args:
        file (str): Path to the file that contains the characters.
    """
    with open(file) as f:
        vocab = f.read()
        vocab = vocab.strip("\r\n\t")
        return list(vocab)


def replace_at(s: str, i: int, r: str) -> str:
    """Create a new string with one replaced character.

    Args:
        s (str): the string to be replaced.
        i (int): the replace position.
        r (str): the replacement character.

    Returns:
        The new string.
    """
    return f"{s[:i]}{r}{s[i+1:]}"


def sanitize(s: str, valid_chars: List[str], replacement=None) -> str:
    """Sanitize the string before encoding.

    This function removes out-of-vocabulary characters and
    replaces characters with similar looks but have different
    code points.

    For some out-of-vocabulary characters, if the de-unicode version
    of that characters is valid, the de-unicode version is used and
    the character is not removed from the final string.

    Args:
        s (str): The string to be sanitized.
        valid_chars (List[str]): List of valid characters.
        replacement (optional): A dictionary of character replacement.
            If not specified, it will be set to :attr:`.DEFAULT_REPACEMENT`.

    Returns:
        The sanitized string.
    """
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
    """Base vocabulary interface."""

    def __init__(self, chars: List[str]):
        """Initialize the vocabulary.

        Args:
            chars (List[str]): List of characters.
        """
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
        """Length of the vocabulary, including the special characters."""
        return len(self.all_chars)

    def batch_decode(self, batch_ids_arr: List[List[int]]):
        """Decode a batch of indices.

        Args:
            batch_ids_arr (List[List[int]]): Batch of character index list.

        Returns:
            List of decoded strings.
        """
        texts = [self.decode(ids) for ids in batch_ids_arr]
        return texts

    def batch_encode(self, texts):
        """Encode a list of strings.

        Args:
            texts (List[str]): List of strings to be encoded.

        Returns:
            List of character index lists.
        """
        ids = [self.encode(c) for c in texts]
        return ids

    def __str__(self):
        """Pretty format the vocab."""
        lines = [
            "Chars: " + "".join(self.chars),
            "Special: " + ", ".join(self.special_tokens),
        ]
        return "\n".join(lines)

    def __repr__(self):
        """Pretty format the vocab."""
        return str(self)

    @abstractmethod
    def get_special_tokens(self):
        """Returns a list of special token names.

        Sub-classes of the :class:`Vocab` class need to
        implement this method themselves. The first special
        tokens should be a padding-like token.

        Returns:
            tokens (List[str]): List of special tokens.
        """
        ...

    @abstractmethod
    def decode(self, seq: List[int]) -> str:
        """Convert a list of character ids to a string.

        The sub-classes in :class:`Vocab` need to implement
        this method themselves.

        Args:
            seq (List[int]): List of character ids.

        Returns:
            The decode string.
        """
        ...

    @abstractmethod
    def encode(self, seq: str, max_length: Optional[int] = None) -> List[int]:
        """Convert a string to a list of supported character indices.

        The sub-classes in :class:`Vocab` need to implement
        this method themselves.

        Args:
            seq (str): The string to be encoded.
            max_length (Optional[int]): The maximum length, if specified,
                the index list will be truncated to this length. Default is
                None. This option is not used in the rewrite.

        Returns:
            The list of character indices.
        """
        ...


class VocabS2S(Vocab):
    """Vocab for sequence-to-sequence (S2S) encoding.

    See torch's S2S tutorial for more information.
    Be warned that the models that use this type of
    coding is prone to hallucination.
    """

    def get_special_tokens(self):
        """Returns S2S specific special characters.

        Special characters for this vocabulary includes:
        - padding,
        - sos (start of sequence),
        - eos (end of sequence),
        - other (unknown character).
        """
        return ["pad", "sos", "eos", "other"]

    def encode(self, chars: str, max_length: Optional[int] = None) -> List[int]:
        """Encode sequence-to-sequence style.

        The input string is first sanitized before encoding.
        The encoded sequence replaces every characters with its character
        indices, and then the sos token is inserted at the beginning,
        the eos token is inserted at the end.
        """
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

    def decode(self, ids: List[int]) -> str:
        """Decode sequence-to-sequence style.

        The index of sos and eos token is used to determine the
        decoding range. Every ids inside that range is converted
        to characters. The obtained characters is concatenated to
        obtain the decoded string.
        """
        last = ids.index(self.eos_id) if self.eos_id in ids else None
        first = ids.index(self.sos_id) if self.sos_id in ids else 0
        if last is not None:
            ids = ids[first:last]
        else:
            ids = ids[first:]
        sent = "".join([self.index2char[i] for i in ids if self.is_normal_id(i)])
        return sent.strip(" ")


class VocabCTC(Vocab):
    """Connectionist temporal classification (CTC) style vocabulary.

    The model that use this type of vocab may converge very slowly, but
    the model is rigid and not prone to hallucination.

    See also: https://distill.pub/2017/ctc/
    """

    def get_special_tokens(self):
        """Returns special CTC tokens.

        The only special token in CTC vocab is the blank character.
        """
        return ["blank"]

    def encode(self, chars):
        """Encode a string CTC style.

        The input string is first sanitized. After that,
        every characters is converted to their respective ids.
        If two consecutive ids are the same, a blank character id
        is inserted between them.
        """
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
        """Perform decoding CTC style using the greedy algorithm.

        If two consecutive ids are the same, they are collapsed in to
        one id. The blank id is then removed from the result id list.
        Finally, the result character id list is converted to character
        and concatenated to obtain the decoded string.

        The original author of VietOCR observed that OCR typically
        only has one correct output, therefore beamsearch for OCR
        is kind of useless.
        """
        collapsed = [self.index2char[i] for i, _ in groupby(ids) if i != self.blank_id]
        return "".join(collapsed).strip(" ")
