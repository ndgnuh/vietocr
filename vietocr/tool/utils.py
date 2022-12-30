import os
import gdown
import yaml
import numpy as np
import uuid
import requests


def annotation_uuid(annotation_file):
    with open(annotation_file, "r", encodng="utf-8") as f:
        return uuid.uuid5(uuid.RFC_4122, f.read().strip())


def download_weights(id_or_url, cached=None, md5=None, quiet=False):
    if id_or_url.startswith('http'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?id={}'.format(id_or_url)

    return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)


def download_config(id):
    url = 'https://raw.githubusercontent.com/ndgnuh/vietocr/master/config/{}'.format(
        id)
    r = requests.get(url)
    config = yaml.safe_load(r.text)
    return config


def compute__accuracies(predictions, ground_truths):
    good = [1 if prediction == ground_truth else 0
            for prediction, ground_truth in zip(predictions, ground_truths)]
    return sum(good) / len(good)


def compute_full_sequence_accuracies(predictions, ground_truths):
    good = [1 if prediction == ground_truth else 0
            for prediction, ground_truth in zip(predictions, ground_truths)]
    return sum(good) / len(good)


def compute_perchar_accuracies(predictions, ground_truths):
    good = [fuzz.ratio(prediction, ground_truth) / 100
            for prediction, ground_truth in zip(predictions, ground_truths)]
    return sum(good) / len(good)


def compute_accuracy(ground_truth, predictions, mode='full_sequence'):
    """
    Computes accuracy
    :param ground_truth:
    :param predictions:
    :param display: Whether to print values to stdout
    :param mode: if 'per_char' is selected then
                 single_label_accuracy = correct_predicted_char_nums_of_single_sample / single_label_char_nums
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
                 if 'full_sequence' is selected then
                 single_label_accuracy = 1 if the prediction result is exactly the same as label else 0
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
    :return: avg_label_accuracy
    """
    if mode == 'per_char':

        accuracy = []

        for index, label in enumerate(ground_truth):
            prediction = predictions[index]
            total_count = len(label)
            correct_count = 0
            try:
                for i, tmp in enumerate(label):
                    if tmp == prediction[i]:
                        correct_count += 1
            except IndexError:
                continue
            finally:
                try:
                    accuracy.append(correct_count / total_count)
                except ZeroDivisionError:
                    if len(prediction) == 0:
                        accuracy.append(1)
                    else:
                        accuracy.append(0)
        avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    elif mode == 'full_sequence':
        try:
            correct_count = 0
            for index, label in enumerate(ground_truth):
                prediction = predictions[index]
                if prediction == label:
                    correct_count += 1
            avg_accuracy = correct_count / len(ground_truth)
        except ZeroDivisionError:
            if not predictions:
                avg_accuracy = 1
            else:
                avg_accuracy = 0
    else:
        raise NotImplementedError(
            'Other accuracy compute mode has not been implemented')

    return avg_accuracy
