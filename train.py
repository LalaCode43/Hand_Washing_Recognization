from lib import *
from utils import *
from imagetransform import *
from mydataset import *
from config import *


def train():
     
    trainvalset = make_data_path_list(phase='trainval')
    # calculate number images of train, val
    total_trainval_len = len(trainvalset)
    train_len = int(0.8*total_trainval_len)
    val_len = total_trainval_len - train_len
    print('Total images of trainval: ', total_trainval_len)
    print('Train: ', train_len)
    print('Validation: ', val_len)

    # split 80% for train and 20% for validation
    trainset, valset = data.random_split(trainvalset, (train_len, val_len),generator=torch.Generator().manual_seed(4))
    
    # list of paths to images of train and val dataset
    train_list = [file for file in trainset]
    val_list = [file for file in valset]

    train_dataset = MyDataset(file_list=train_list, transform=ImageTransform(resize, mean, std), phase='train')
    val_dataset = MyDataset(file_list=val_list, transform=ImageTransform(resize, mean, std), phase='val')

    train_dataloader = data.DataLoader(train_dataset,  batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset,  batch_size, shuffle=False)

    dataloader_dict = {
        'train': train_dataloader,
        'val': val_dataloader
        }
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # load pretrained model MobileNet_V2
    use_pretrained = True
    net = models.mobilenet_v2(pretrained=use_pretrained)

    # freeze features extrator
    for name, params in net.named_parameters():
        params.requires_grad = False

    # modify fully connected layer for 7 class to classify
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
    

    # net = load_model(net, '../weights1/mobilenet_5.pth')


    # move network to device
    net = net.to(device)
    torch.backends.cudnn.benchmark = True

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    # train processing
    train_model(net, dataloader_dict, criterion, optimizer, scheduler, 3, device)

    # unfreeze some high layers
    idx = 0 
    for name, params in net.named_parameters():
        if idx < 99: # train from features.12.conv.0.0.weight
            params.requires_grad = False
        else:
            params.requires_grad = True
        idx += 1
        
    # move network to device
    net = net.to(device)
    torch.backends.cudnn.benchmark = True

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    # train processing
    train_model(net, dataloader_dict, criterion, optimizer, scheduler, num_epochs, device)

if __name__ == '__main__':
    train()