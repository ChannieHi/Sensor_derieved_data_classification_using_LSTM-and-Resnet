import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.model import MLSTMfcn, Resnet
from utils_LSTM import train, load_datasets
from utils_Resnet import train, load_datasets
from src.constants import NUM_CLASSES, MAX_SEQ_LEN, NUM_FEATURES
from utils_Resnet import train, load_datasets

def main():
    wandb.login()
    name = " "   ## Give your own trial name
    wandb.init(project=" ", name=name)  ## Give your own project name

    dataset = args.dataset
    assert dataset in NUM_CLASSES.keys()

    train_dataset, val_dataset, _ = load_datasets(dataset_name=dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    ########### With Win OS ########### 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ############## With Mac ############
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
    print("Device: {}".format(device))

    model = MLSTMfcn(num_classes=NUM_CLASSES[dataset], 
                               max_seq_len=MAX_SEQ_LEN[dataset], 
                               num_features=NUM_FEATURES[dataset])
    # model = Resnet(in_channels=20, num_classes=7)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    criterion = nn.NLLLoss()                    #LSTM
    #criterion = torch.nn.CrossEntropyLoss()    #Resnet
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    train(model, train_loader, val_loader, 
          criterion, optimizer, scheduler,
          epochs=args.epochs, device=device, run_name=args.name)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--name", type=str, default="model_mlstm_fcn")
    p.add_argument("--dataset", type=str, default="npy")
    args = p.parse_args()
    main()