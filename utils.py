import pandas as pd
import io
import numpy as np


def load_vectors(embedding_file):
    fin = io.open(embedding_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def read_metadata(csv_file_path):
    csv_data = pd.read_csv(csv_file_path, encoding='latin1')
    audio_fnames = list(csv_data['file_name'])
    questions = list(csv_data['QuestionText'])
    answers = list(csv_data['answer'])
    return audio_fnames, questions, answers


def classification_accuracy(pred, ground_truth):
    n_samples = pred.shape[0]
    x = pred - ground_truth
    n_wrong_predictions = np.count_nonzero(x)
    accuracy = (n_samples - n_wrong_predictions) / n_samples
    return accuracy
