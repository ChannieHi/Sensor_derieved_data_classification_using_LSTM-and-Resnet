import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from src.model import MLSTMfcn, Resnet
from utils_LSTM import validation, load_datasets
from utils_Resnet import validation, load_datasets
from src.constants import NUM_CLASSES, MAX_SEQ_LEN, NUM_FEATURES
from sklearn.metrics import precision_score, recall_score, f1_score


def main():
    dataset = args.dataset

    assert dataset in NUM_CLASSES.keys()

    _, _, test_dataset = load_datasets(dataset_name=dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    pred_list = []                                     
    target_list = [] 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    print("Device: {}".format(device))

    model = MLSTMfcn(num_classes=NUM_CLASSES[dataset], 
                               max_seq_len=MAX_SEQ_LEN[dataset], 
                               num_features=NUM_FEATURES[dataset])
    # model = Resnet(in_channels=20, num_classes=7)
    model.load_state_dict(torch.load('weights/'+args.weights))
    model.to(device)

    criterion = nn.NLLLoss()                    #LSTM
    #criterion = torch.nn.CrossEntropyLoss()    #Resnet
    
    test_loss, test_acc, pred, target = validation(model, test_loader, criterion, device)
    pred_list.append(pred)
    target_list.append(target)
    pred_list = np.concatenate(pred_list, axis=0)
    target_list = np.concatenate(target_list, axis=0)
    precision = precision_score(pred_list, target_list, average='weighted')
    f1 = f1_score(pred_list, target_list, average='weighted')
    recall = recall_score(pred_list, target_list, average='weighted')

    print("Test loss: {:.6f}.. Test Accuracy: {:.2f}%".format(test_loss/len(test_loader), (test_acc/len(test_loader))*100))
    print("f1: {:3.2f}.. precision: {:3.2f}%".format(f1, precision))
    print("recall: {:3.2f}".format(recall))

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--weights", type=str, default="model_mlstm_fcn.pt")
    p.add_argument("--dataset", type=str, default="npy")
    args = p.parse_args()
    main()