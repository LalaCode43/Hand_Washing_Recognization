from lib import *
from utils import *
from imagetransform import *
from mydataset import *
from config import *


def train():
    trainvalset = make_data_path_list(phase='trainval')
    trainset, valset = data.random_split(trainvalset, (568183, 164328),generator=torch.Generator().manual_seed(4))
    
    # list of paths to images of train and val dataset
    train_list = [file for file in trainset]
    val_list = [file for file in valset]

    train_dataset = MyDataset(file_list=train_list, transform=ImageTransform(resize, mean, std), phase='train')
    val_dataset = MyDataset(file_list=val_list, transform=ImageTransform(resize, mean, std), phase='val')

    train_dataloader = data.DataLoader(train_dataset,  batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset,  batch_size, shuffle=True)

    dataloader_dict = {
        'train': train_dataloader,
        'val': val_dataloader
        }
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load pretrained model MobileNet_V2
    use_pretrained = True
    net = models.mobilenet_v2(pretrained=use_pretrained)

    # modify fully connected layer for 7 class to classify
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features=in_features, out_features=7)

    # move network to device
    net = net.to(device)
    torch.backends.cudnn.benchmark = True

    # freeze low layers 
    ct = 0
    for child in net.children():
        ct += 1 
        for name, params in child.named_parameters():
            if int(name.split('.')[0]) < 8 and ct == 1:
                params.requires_grad = False

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.008)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    # train processing
    train_model(net, dataloader_dict, criterion, optimizer, scheduler, num_epochs, device)


if __name__ == '__main__':
    train()