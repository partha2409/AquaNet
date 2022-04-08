import torch.nn as nn
import torch
import torch.nn.functional as F


class AquaNet(nn.Module):
    def __init__(self, hp):
        super(AquaNet, self).__init__()
        self.audio_lstm = nn.LSTM(input_size=hp['audio_input_size'], hidden_size=hp['audio_lstm_hidden_size'], num_layers=hp['audio_lstm_n_layers'],
                                  batch_first=True, dropout=hp['audio_lstm_dropout'], bidirectional=hp['audio_bidirectional'])

        self.text_lstm = nn.LSTM(input_size=hp['text_input_size'], hidden_size=hp['text_lstm_hidden_size'], num_layers=hp['text_lstm_n_layers'],
                                 batch_first=True, dropout=hp['text_lstm_dropout'], bidirectional=hp['text_bidirectional'])

        self.dense_layer1 = nn.Linear(in_features=hp['dense1_input'], out_features=hp['n_dense1_units'])
        self.dense_layer2 = nn.Linear(in_features=hp['n_dense1_units'], out_features=hp['n_dense2_units'])
        self.classification_layer = nn.Linear(in_features=hp['n_dense2_units'], out_features=hp['n_classes'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, audio_feat_in, text_feat_in):

        audio_feat_out, _ = self.audio_lstm(audio_feat_in)
        text_feat_out, _ = self.text_lstm(text_feat_in)

        audio_feat_out = audio_feat_out[:, -1, :]
        text_feat_out = text_feat_out[:, -1, :]

        merge_feat = torch.cat([audio_feat_out, text_feat_out], dim=-1)

        dense_out = F.relu(self.dense_layer1(merge_feat))
        dense_out = F.relu(self.dense_layer2(dense_out))
        logits = self.sigmoid(self.classification_layer(dense_out))
        return logits


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, prediction, target):
        prediction = torch.reshape(prediction, [-1])
        target = torch.reshape(target, [-1])
        loss = self.bce_loss(prediction, target)
        return loss
