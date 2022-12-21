import unidecode


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


class Vocab():
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3
        self.other = 4

        self.chars = chars

        self.c2i = {c: i+5 for i, c in enumerate(chars)}

        self.i2c = {i+5: c for i, c in enumerate(chars)}

        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'
        self.i2c[4] = '<unk>'

    def encode(self, chars, max_length=None):
        vocab = self.chars
        chars = unidecode_string(chars, vocab)
        seq = [self.c2i.get(c, self.other) for c in chars]
        if max_length is not None:
            n = len(seq)
            seq = seq[:max_length - 2]
            seq = seq + [self.pad] * (max_length - 2 - n)
        return [self.go] + seq + [self.eos]

    def decode(self, ids):
        last = ids.index(self.eos) if self.eos in ids else None
        first = ids.index(self.go) if self.go in ids else 0
        if last is not None:
            ids = ids[first:last]
        else:
            ids = ids[first:]
        sent = ''.join([self.i2c[i] for i in ids if i > 4])
        return sent

    def __len__(self):
        return len(self.c2i) + 5

    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars
