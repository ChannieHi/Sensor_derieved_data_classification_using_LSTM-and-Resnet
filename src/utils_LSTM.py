import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics._plot.roc_curve import roc_curve

def validation(model, testloader, criterion, device='cpu'):
    val_acc = 0
    val_loss = 0
    for inputs, labels, seq_lens in testloader:
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        val_output = model.forward(inputs, seq_lens)
        val_loss += criterion(val_output, labels).item()

        y_pred_tag = (torch.tanh(torch.exp(val_output))).max(1)[1]
        correct_results_sum = (y_pred_tag == labels).sum().float()  
        
        val_acc += correct_results_sum/labels.shape[0] 
        # ## Calculating the accuracy 
        # # Model's output is log-softmax, take exponential to get the probabilities
        # ps = torch.exp(output)
        # # Class with highest probability is our predicted class, compare with true label
        # equality = (labels.data[-1] == ps.max(1)[1]).sum().float()  
        # # Accuracy is number of correct predictions divided by all predictions, just take the mean
        # acc += equality.type_as(torch.FloatTensor()).mean()

    return val_loss, val_acc, y_pred_tag.detach().cpu().numpy(), val_labels.detach().cpu().numpy()


def train(model, trainloader, validloader, criterion, optimizer, scheduler,
          epochs=100, device='cpu', run_name='model_mlstm_fcn'):
    print("Training started on device: {}".format(device))

    valid_loss_min = np.Inf # track change in validation loss

    
    for e in range(epochs):
        train_loss = 0.0
        train_acc = 0
        pred_list = []                                     
        target_list = []  
        model.train()
        steps = 0
        for inputs, labels, seq_lens in trainloader:

            inputs = inputs.float()
            inputs, labels = inputs.to(device),labels.to(device)
            
            optimizer.zero_grad()
            
            output = model.forward(inputs, seq_lens)
            loss = criterion(output, labels)
            
            y_pred_tag = (torch.tanh(torch.exp(output))).max(1)[1]
            correct_results_sum = (y_pred_tag == labels).sum().float()  

            acc = correct_results_sum/labels.shape[0] 
            acc = torch.round(acc)   
    
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()
            train_acc += acc.detach().item()

            steps += 1

        
        model.eval()
        
        with torch.no_grad():
            val_loss, val_acc, pred, target = validation(model, validloader, criterion, device)
            pred_list.append(pred)
            target_list.append(target)

        scheduler.step() 
        print(f"Epoch: {e+1}/{epochs} ")
        print('-' * 20),
        print("Training Accuracy: {:.2f}%".format((train_acc/len(trainloader))*100),f"Training Loss: {train_loss/len(trainloader):.3f}")
        print("Val Accuracy: {:.2f}%".format((val_acc/len(validloader))*100), "Val Loss: {:.6f} ".format(val_loss/len(validloader)))
        print('\n')

        pred_list = np.concatenate(pred_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)
        precision = precision_score(pred_list, target_list, average='micro')
        f1 = f1_score(pred_list, target_list, average='micro')
        recall = recall_score(pred_list, target_list, average='micro')

        
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

    X_train =  torch.from_numpy(np.load(data_path+'X_train.npy').astype('float32'))
    X_val =  torch.from_numpy(np.load(data_path+'X_val.npy').astype('float32'))
    X_test =  torch.from_numpy(np.load(data_path+'X_test.npy').astype('float32'))

    y_train = torch.from_numpy(np.load(data_path+'y_train.npy')).long()
    y_val = torch.from_numpy(np.load(data_path+'y_val.npy')).long()
    y_test = torch.from_numpy(np.load(data_path+'y_test.npy')).long()

    seq_len_train = torch.from_numpy(np.load(data_path+'train_seq.npy').astype('float32'))
    seq_len_test = torch.from_numpy(np.load(data_path+'test_seq.npy').astype('float32'))
    seq_len_val = torch.from_numpy(np.load(data_path+'val_seq.npy').astype('float32'))
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, seq_len_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val, seq_len_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test, seq_len_test)

    return train_dataset, val_dataset, test_dataset