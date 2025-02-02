import torch
import numpy as np
import math
import gdown
from PIL import Image
from torch.nn.functional import log_softmax, softmax

from vietocr.model.transformerocr import VietOCR
from vietocr.model.vocab import VocabS2S, VocabCTC
from vietocr.model.beam import Beam
from . import utils


def loosely_load_state_dict(model, sd):
    errors = []
    orig_sd = model.state_dict()
    for key, value in sd.items():
        if key not in orig_sd:
            errors.append(key)
            continue

        orig_value = orig_sd[key]
        if orig_value.shape != value.shape:
            errors.append(key)
            continue

        orig_sd[key] = value

    errors = '\n'.join([f'\t{k}' for k in errors])
    print(f"Mismatch keys:\n{errors}")
    model.load_state_dict(orig_sd)
    return model


def batch_translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: NxCxHxW
    model.eval()
    device = img.device
    sents = []

    with torch.no_grad():
        src = model.cnn(img)
        print(src.shap)
        memories = model.transformer.forward_encoder(src)
        for i in range(src.size(0)):
            #            memory = memories[:,i,:].repeat(1, beam_size, 1) # TxNxE
            memory = model.transformer.get_memory(memories, i)
            sent = beamsearch(memory, model, device, beam_size,
                              candidates, max_seq_length, sos_token, eos_token)
            sents.append(sent)

    sents = np.asarray(sents)

    return sents


def translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: 1xCxHxW
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)  # TxNxE
        sent = beamsearch(memory, model, device, beam_size,
                          candidates, max_seq_length, sos_token, eos_token)

    return sent


def beamsearch(memory, model, device, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # memory: Tx1xE
    model.eval()

    beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates,
                ranker=None, start_token_id=sos_token, end_token_id=eos_token)

    with torch.no_grad():
        #        memory = memory.repeat(1, beam_size, 1) # TxNxE
        memory = model.transformer.expand_memory(memory, beam_size)

        for _ in range(max_seq_length):

            tgt_inp = beam.get_current_state().transpose(0, 1).to(device)  # TxN
            decoder_outputs, memory = model.transformer.forward_decoder(
                tgt_inp, memory)

            log_prob = log_softmax(
                decoder_outputs[:, -1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)

    return [1] + [int(i) for i in hypothesises[0][:-1]]


@torch.no_grad()
def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    model.eval()
    device = img.device
    batch_size = len(img)

    feat = model.cnn(img)
    encoder_outputs, hidden = model.transformer.encoder(feat)
    logits = []
    input = torch.tensor([sos_token] * batch_size, device=device)
    translated = []
    for i in range(max_seq_length):
        prob, hidden, _ = model.transformer.decoder(
            input,
            hidden,
            encoder_outputs
        )
        input = torch.argmax(prob, dim=-1)
        translated.append(input)
        logits.append(prob)

    logits = torch.log_softmax(torch.stack(logits, dim=0), dim=-1)
    probs, _ = logits.topk(k=1, dim=-1)

    translated = torch.stack(translated, dim=1)
    probs = (probs.sum(dim=0) / (probs > 0).count_nonzero(dim=0)).squeeze(-1)

    return translated, probs, logits


def translate_old(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: BxCXHxW"
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token]*len(img)]
        char_probs = [[1]*len(img)]

        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)

#            output = model(img, tgt_inp, tgt_key_padding_mask=None)
#            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)
            output = output.to('cpu')

            values, indices = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence > 3)
        char_probs = np.sum(char_probs, axis=-1)/(char_probs > 0).sum(-1)

    return translated_sentence, char_probs


def load_weights(path, reset_cache=False):
    if path.startswith('http'):
        if reset_cache:
            download_fn = gdown.download
        else:
            download_fn = gdown.cached_download
        weights = torch.load(download_fn(path), map_location="cpu")
    else:
        weights = torch.load(path, map_location="cpu")
    return weights


def build_model(config, move_to_device=True, reset_cache=False):
    if config['type'] == 's2s':
        vocab = VocabS2S(config['vocab'])
    elif config['type'] == 'ctc':
        vocab = VocabCTC(config['vocab'])

    model = VietOCR(len(vocab),
                    config['backbone'],
                    config['cnn'],
                    config['transformer'],
                    config['seq_modeling'],
                    stn=config.get('stn', None))
    if 'weights' in config:
        weights = load_weights(config['weights'], reset_cache=reset_cache)
        loosely_load_state_dict(model, weights)

    if move_to_device:
        device = utils.get_device(config.get('device', None))
        model = model.to(device)

    return model, vocab


def resize(
    w,
    h,
    expected_height,
    image_min_width,
    image_max_width,
    strict=False,
    align_width: int = 10
):
    new_w = int(expected_height * float(w) / float(h))
    new_w = math.ceil(new_w/align_width)*align_width
    if strict:
        assert new_w <= image_max_width, f"Image too wide: {new_w}"
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width, **k):
    img = image.convert('RGB')

    w, h = img.size
    new_w, image_height = resize(
        w, h, image_height, image_min_width, image_max_width, **k)

    img = img.resize((new_w, image_height), Image.Resampling.LANCZOS)

    img = np.asarray(img).transpose(2, 0, 1)
    img = img/255
    return img


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img


def predict(filename, config):
    img = Image.open(filename)
    img = process_input(img)

    img = img.to(config['device'])

    model, vocab = build_model(config)
    s = translate(img, model)[0].tolist()
    s = vocab.decode(s)

    return s
