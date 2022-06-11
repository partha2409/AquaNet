import torch
import numpy as np
from torch.utils.data import DataLoader
from model import AquaNet, BinaryCrossEntropyLoss, CrossEntropyLoss
from data_generator import DataGenerator
from conifg import data_config, model_config
import time
import matplotlib.pyplot as plt
import pickle
import utils
import os


def train(pre_trained=None):

    # create dir to save trained models and loss graphs
    reference = model_config['net_type'] + str(time.strftime("_%Y%m%d_%H%M%S"))
    checkpoints_folder = os.path.join(model_config["output_dir"], 'checkpoints', reference)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # save hyper parameters config
    pickle_file_location = os.path.join(checkpoints_folder, "hp.pkl")
    pickle_file = open(pickle_file_location, "wb")
    pickle.dump(model_config, pickle_file)
    pickle_file.close()

    # create data iterator
    train_data_set = DataGenerator(data_config)
    iterator = DataLoader(dataset=train_data_set, batch_size=model_config['batch_size'], num_workers=model_config['num_workers'], pin_memory=True, shuffle=True, drop_last=True)

    val_set = DataGenerator(data_config, mode='val')
    val_set_iterator = DataLoader(dataset=val_set, batch_size=model_config['batch_size'], num_workers=model_config['num_workers'], pin_memory=True, shuffle=False, drop_last=True)
    # create model and loss

    model = AquaNet(model_config).to(device)
    if 'binary' in data_config['train_metadata_path']:
        loss = BinaryCrossEntropyLoss().to(device)
    else:
        loss = CrossEntropyLoss().to(device)
    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=model_config['learning_rate'])
    start_epoch = 0
    # load pre trained model

    if pre_trained is not None:
        ckpt = torch.load(pre_trained)
        model.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1

    # init loss arrays
    classification_loss = np.zeros(model_config['num_epochs'])
    train_accuracy = np.zeros(model_config['num_epochs'])
    val_accuracy = np.zeros(model_config['num_epochs'])

    # training loop
    for epoch in range(start_epoch, model_config['num_epochs']):
        c_loss = 0
        acc = 0
        for i, (audio_feat, text_feat, label) in enumerate(iterator):
            audio_feat = audio_feat.to(device, dtype=torch.float)
            text_feat = text_feat.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)

            optimizer.zero_grad()
            logits = model(audio_feat, text_feat)
            l = loss(logits, label)
            l.backward()
            optimizer.step()
            c_loss += l.item()

            # calc accuracy
            pred = torch.round(logits)
            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            acc += utils.classification_accuracy(pred, label)

        # average loss per epoch
        classification_loss[epoch] = c_loss/(i+1)
        # average accuracy per epoch
        train_accuracy[epoch] = acc/(i+1)

        print("epoch = {}, average classification loss ={}".format(epoch, classification_loss[epoch]))
        print("epoch = {}, Training accuracy ={}".format(epoch, train_accuracy[epoch]))

        with torch.no_grad():
            val_acc = 0
            for i, (audio_feat, text_feat, label) in enumerate(val_set_iterator):
                audio_feat = audio_feat.to(device, dtype=torch.float)
                text_feat = text_feat.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)
                logits = model(audio_feat, text_feat)

                # calc accuracy
                pred = torch.round(logits)
                pred = pred.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                val_acc += utils.classification_accuracy(pred, label)

        val_accuracy[epoch] = val_acc/(i+1)
        print("epoch = {},  Validation set accuracy ={}".format(epoch, val_accuracy[epoch]))
        print('***********************************************************')

        # plot accuracy curves and save model
        plt.plot(range(1, len(train_accuracy)+1), train_accuracy, 'b-', label=" Train Accuracy")
        plt.plot(range(1, len(val_accuracy)+1), val_accuracy, 'r-', label="Validation Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='best')
        plt.savefig(checkpoints_folder + "/accuracy.jpeg", bbox_inches="tight")
        plt.clf()
        if (epoch+1) % 10 == 0: # save every 10th epoch
            net_save = {'net': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch}
            torch.save(net_save, checkpoints_folder + "/aquanet_epoch{}.pth".format(epoch))


if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    pre_trained_model_path = None    # provide path to .pth to continue training if you have checkpoint files.
    train(pre_trained=pre_trained_model_path)
