import unidecode
from abc import abstractmethod, ABCMeta


def read_vocab_file(file):
    with open(file) as f:
        vocab = f.read()
        vocab = vocab.strip("\r\n\t")
        return list(vocab)


def replace_at(s, i, sub):
    left, right = s[:i], s[(i + 1):]
    return f"{left}{sub}{right}"


def unidecode_string(s, vocab, max_iteration=10):
    custom_mapping = {"−": "-", "Ð": "Đ"}
    prev_i = 0
    n = len(s)
    prev_s = None
    # print("==============================")
    # print("String:", s)
    while True:
        # print(s)
        # If previous string is the same, return
        if prev_s == s:
            return s

        # Loop through and replace
        all_in_vocab = True
        for i in range(prev_i, n):
            c = s[i]
            # Replace with char inside vocab
            if c not in vocab:
                all_in_vocab = False
                # print('- char', c)
                replacement = custom_mapping.get(c, unidecode.unidecode(c))
                if any([r not in vocab for r in replacement]):
                    print(
                        f"Warning: replacement {replacement} not in vocab, using blank")
                    replacement = ""
                prev_s = s
                s = replace_at(s, i, replacement)

                # Store index to loop efficiently
                prev_i = i
                continue

        if all_in_vocab:
            break
    # print("Result:", s)
    return s


class Vocab(metaclass=ABCMeta):
    @classmethod
    def from_file(cls, file):
        return cls(read_vocab_file(file))

    def __init__(self, chars):
        special_tokens = self.get_special_tokens()
        for i, token in enumerate(special_tokens):
            setattr(self, f"{token}_id", i)

        self.chars = list(chars)
        self.special_tokens = [f"<{token}>" for token in special_tokens]
        vocabs = list(self.special_tokens) + list(chars)
        self.c2i = {c: i for i, c in enumerate(vocabs)}
        self.i2c = {i: c for i, c in enumerate(vocabs)}
        self.vocabs = vocabs

        self.num_special_tokens = len(special_tokens)

    def is_normal_id(self, i):
        return i >= self.num_special_tokens

    def is_special_id(self, i):
        return i < self.num_special_tokens

    def __len__(self):
        return len(self.vocabs)

    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def batch_encode(self, arr, max_length=None):
        ids = [self.encode(c, max_length=max_length) for c in arr]
        return ids

    def __str__(self):
        lines = [
            "Chars: " + ''.join(self.chars),
            "Special: " + ', '.join(self.special_tokens)
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
        chars = unidecode_string(chars, vocab)
        seq = [self.c2i.get(c, self.other_id) for c in chars]
        if max_length is not None:
            n = len(seq)
            seq = seq[:max_length - 2]
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
        sent = ''.join([self.i2c[i] for i in ids if self.is_normal_id(i)])
        return sent


class VocabCTC(Vocab):
    def get_special_tokens(self):
        return ["blank"]

    def encode(self, chars, max_length=None):
        # chars = unidecode_string(chars, self.chars)
        ids = []
        prev = None
        for c in chars:
            if prev == c:
                ids.append(self.blank_id)
            ids.append(self.c2i.get(c, self.blank_id))
            prev = c
        if max_length:
            n = len(ids)
            assert n <= max_length
            ids.extend([self.blank_id] * (max_length - n))
        return ids

    # greedy ctc decode
    def decode(self, ids):
        from itertools import groupby
        collapsed = [self.i2c[i] for i, _ in groupby(ids)
                     if i != self.blank_id]
        return ''.join(collapsed)
