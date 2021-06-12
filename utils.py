import os.path
import torch
from lib import *
from config import *


def make_data_path_list(phase='trainval'):
    # '.data/trainval/*/*.jpg'
    target_path = os.path.join(root_path, phase, '*/*.jpg')
    
    path_list = list()
    
    for path in glob.glob(target_path):
        path_list.append(path)
    
    return path_list


def train_model(net, dataloader_dict, criterion, optimizer, scheduler, num_epochs, device):
    history = []
    logs = []
    train_data_size = len(dataloader_dict['train'].dataset)
    val_data_size = len(dataloader_dict['val'].dataset)
    for epoch in range(num_epochs):
        print("Epoch: {}/{}".format(epoch + 1, num_epochs))

        # Set to training mode
        net.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        val_loss = 0.0
        val_acc = 0.0

        for i, (inputs, labels) in tqdm(enumerate(dataloader_dict['train'])):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = net(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()
            scheduler.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)

            # Compute the accuracy
            ret, preds = torch.max(outputs.data, 1)
            corrects = preds.eq(labels.data.view_as(preds))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(corrects.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            net.eval()

            # Validation loop
            for j, (inputs, labels) in tqdm(enumerate(dataloader_dict['val'])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = net(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                val_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, preds = torch.max(outputs.data, 1)
                corrects = preds.eq(labels.data.view_as(preds))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(corrects.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                val_acc += acc.item() * inputs.size(0)

        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = val_loss / val_data_size
        avg_valid_acc = val_acc / val_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        print('Epoch: {:03d}'.format(epoch+1))
        print('Training: Loss - {:.4f}, Accuracy - {:.4f}%'.format(avg_train_loss, avg_train_acc * 100))
        print('Validation : Loss - {:.4f}, Accuracy - {:.4f}%'.format(avg_valid_loss, avg_valid_acc * 100))

        log_epoch = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc':  avg_train_acc,
            'val_loss': avg_valid_loss,
            'val_acc': avg_valid_acc
        }
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("../weights/mobilenet_logs.csv")
        torch.save(net.state_dict(), "../weights/mobilenet_" + str(epoch+1) + ".pth")


def test_model(net, test_dataloader, criterion, device):
    
    test_data_size = len(test_dataloader.dataset)
    test_acc = 0.0
    test_loss = 0.0
    cf_matrix = torch.zeros(6, 6)
    for i, (inputs, labels) in tqdm(enumerate(test_dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        
        with torch.no_grad():
            # set network to eval phase
            net.eval() 
            
            # inputs pass through network
            outputs = net(inputs)
            
            # compute loss 
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()* inputs.size(0)
            
            # predict
            ret, preds = torch.max(outputs.data, dim=1)
            corrects = preds.eq(labels.data.view_as(preds))          
    
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(corrects.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs.size(0)
        
        # compute Confusion Matrix
        for i in range(preds.size(0)):
            cf_matrix[int(labels[i].item()), int(preds[i].item())] += 1
    
            
    # compute mean confusion matrix
    mean_cf_matrix = cf_matrix / cf_matrix.sum(dim=1, keepdim=True)
    
    # Find average training loss and training accuracy
    avg_test_acc = test_acc / test_data_size

    # Find average training loss and training accuracy
    avg_test_loss = test_loss / test_data_size
    
    print('Loss- {:03f}, Accuracy: {:03f}%'.format(avg_test_loss, avg_test_acc*100))
    print('Confusion table:')
    print(mean_cf_matrix)
    
    
def load_model(net, model_path):
    if torch.cuda.is_available(): # if load on GPU
        load_weight = torch.load(model_path)
    else: # else if load on CPU
        load_weight = torch.load(model_path, map_location={'cuda:0': 'cpu'})
    net.load_state_dict(load_weight)
    return net


if __name__ == '__main__':
    trainvalset = make_data_path_list('trainval')
    testset = make_data_path_list('test')
    print(len(trainvalset))
    print(len(testset))