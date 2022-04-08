import os
from torch.utils.data.dataset import Dataset
from utils import load_vectors, read_metadata
import numpy as np


class DataGenerator(Dataset):

    def __init__(self, data_config, mode='train'):
        super(DataGenerator, self).__init__()

        if mode == 'train':
            self.meta_file = data_config['train_metadata_path']
        else:
            self.meta_file = data_config['val_metadata_path']
        self.feat_dir = data_config['feat_dir']
        self.audio_fnames, self.qs, self.ans = read_metadata(self.meta_file)

        self.word_embedding_path = data_config['pre_trained_word_embeddings_file']
        self.word_embeddings = load_vectors(self.word_embedding_path)  # dict of all the {'word': [vector]} pairs

    def __getitem__(self, item):
        audio_feat = self.load_audio_features(item)
        question_text = self.qs[item]
        answer_text = self.ans[item]
        question_embedding = self.get_word_embeddings(question_text)

        if answer_text == 'YES':
            label = 0 
        else:
            label = 1
        return audio_feat, question_embedding, label

    def load_audio_features(self, idx):
        audio_feat_file = self.audio_fnames[idx][:-3] + 'npz'
        data = np.load(os.path.join(self.feat_dir, audio_feat_file))
        return data['embedding']

    def get_word_embeddings(self, input_text):
        words = input_text.split(' ')
        words[-1] = words[-1][:-1]  # removing '?' from the question, repetitive in all the Qs, so adds no value
        text_embedding = []
        for word in words:
            try:
                embedding = self.word_embeddings[word]
            except KeyError:
                continue
            text_embedding.append(embedding)
        return np.array(text_embedding)

    def __len__(self):
        return len(self.qs)
