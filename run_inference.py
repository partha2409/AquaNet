import torch
from model import AquaNet
import utils
import pickle
from torch.utils.data import DataLoader
from data_generator import DataGenerator
from conifg import data_config


def run_inference(model_dir, model_path):
    hp_file = model_dir + 'hp.pkl'
    f = open(hp_file, "rb")
    hp = pickle.load(f)

    model = AquaNet(hp).to(device)
    ckpt = torch.load(model_dir+model_path, map_location=device)
    model.load_state_dict(ckpt['net'])

    dataset = DataGenerator(data_config, mode='test')
    test_iterator = DataLoader(dataset=dataset, batch_size=1, num_workers=hp['num_workers'], pin_memory=True,
                          shuffle=False, drop_last=True)

    with torch.no_grad():
        test_acc = 0
        if hp['n_classes'] != 1:
            top5_test_acc = 0
            top10_test_acc = 0
        for i, (audio_feat, text_feat, label) in enumerate(test_iterator):
            audio_feat = audio_feat.to(device, dtype=torch.float)
            text_feat = text_feat.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)
            logits = model(audio_feat, text_feat)

            # calc accuracy
            if hp['n_classes'] == 1:
                pred = torch.round(logits)
                pred = pred.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                test_acc += utils.binary_classification_accuracy(pred, label)

            else:
                logits = logits.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                test_acc += utils.multiclass_classification_accuracy(logits, label)
                top5_test_acc += utils.multiclass_classification_accuracy(logits, label, k=5)
                top10_test_acc += utils.multiclass_classification_accuracy(logits, label, k=10)

        print("Test set accuracy ={}".format(test_acc/(i+1)))
        if hp['n_classes'] != 1:
            print("Top 5 Test set accuracy ={}".format(top5_test_acc/(i+1)))
            print("Top 10 Test set accuracy ={}".format(top10_test_acc/(i+1)))


if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_dir = None  # "checkpoints/aquanet_20220208_141824/"
    model_path = None  # "aquanet_epoch49.pth"
    run_inference(model_dir, model_path)
