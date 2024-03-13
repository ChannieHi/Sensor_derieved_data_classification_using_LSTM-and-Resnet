import wandb
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics._plot.roc_curve import roc_curve

def validation(model, testloader, criterion, device='cpu'):
    val_acc = 0.0
    val_loss = 0.0
    for val_input, val_label in testloader:
        val_inputs = val_input.to(device)
        val_labels = val_label.to(device)

        val_output = model(val_inputs)

        v_loss = criterion(val_output, val_labels)

        y_pred_tag = val_output.max(1)[1]
        correct_results_sum = (y_pred_tag == val_labels).sum().float()
        val_acc = correct_results_sum/val_labels.shape[0]
        val_acc = torch.round(val_acc * 100)

        val_loss += v_loss
        val_acc += val_acc.item()

    return val_loss, val_acc, y_pred_tag.detach().cpu().numpy(), val_labels.detach().cpu().numpy()


def train(model, trainloader, validloader, criterion, optimizer, scheduler,
          epochs=100, device='cpu', run_name='model_Resnet'):
    
    print("Training started on device: {}".format(device))
    best_acc = 0
                                  
    for epoch in range(epochs):                    
        print(f'Epoch {epoch}/{epochs - 1}')        
        print('-' * 20)
        train_loss = 0.0                             
        train_acc = 0                                                             
        model.train()                                
        iter = 0
        pred_list = []                                     
        target_list = []  
        for inputs, labels in trainloader:         

                inputs = inputs.to(device)
                labels = labels.to(device)     

                optimizer.zero_grad()          

                pred = model(inputs)       
    
                loss = criterion(pred, labels)                  
                y_pred_tag = pred.max(1)[1]    
                                 
                correct_results_sum = (y_pred_tag == labels).sum().float()     
                acc = correct_results_sum/labels.shape[0]                     
                acc = torch.round(acc * 100)                                              

                loss.backward()                                                            
                optimizer.step()                                                    

                # 통계
                train_loss += loss.item()                                                  
                train_acc += acc.item()

                iter +=1                                                

        model.eval()               
        with torch.no_grad():
            val_loss, val_acc, pred, target = validation(model, validloader, criterion, device)


            pred_list.append(pred)
            target_list.append(target)

        scheduler.step()                               
        pred_list = np.concatenate(pred_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)
        precision = precision_score(pred_list, target_list, average='weighted')
        f1 = f1_score(pred_list, target_list, average='weighted')
        recall = recall_score(pred_list, target_list, average='weighted')

        wandb.log({
        "train_loss": train_loss/len(trainloader),
        "val_loss": val_loss/len(validloader),
        "train_acc": (train_acc/len(trainloader)) * 100,
        "val_acc": (val_acc/len(validloader)) * 100,
        'precision':precision,
        'recall':recall,
        'f1':f1
        })

        # save model if validation loss has decreased
        if val_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            val_loss))
            torch.save(model.state_dict(), 'weights/'+run_name+'.pt')
            valid_loss_min = val_loss


def load_datasets(dataset_name='npy'):
    data_path = './datasets/'+dataset_name+'/'

    train_transform = torchvision.transforms.Compose([
                                                transforms.Resize((128,128)),
                                                transforms.ToTensor()
    ])

    val_transform = torchvision.transforms.Compose([
                                                transforms.Resize((128,128)),
                                                transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.ImageFolder(data_path+'train',transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(data_path+'valid', transform=val_transform)
    test_dataset = torchvision.datasets.ImageFolder(data_path+'test', transform=val_transform, target_transform=None)

    return train_dataset, val_dataset, test_dataset