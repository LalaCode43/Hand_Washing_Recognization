from lib import *
from utils import *
from imagetransform import *
from mydataset import *
from config import *


def test():
    test_list = make_data_path_list(phase='test')

    test_dataset = MyDataset(file_list=test_list, transform=ImageTransform(resize, mean, std), phase='val')
   
    test_dataloader = data.DataLoader(test_dataset,  batch_size, shuffle=False)
    
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # load pretrained model MobileNet_V2
    use_pretrained = False
    net = models.mobilenet_v2(pretrained=use_pretrained)

    # modify fully connected layer for 7 class to classify
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)

    # move network to device
    net = net.to(device)
    torch.backends.cudnn.benchmark = True
    
    net = load_model(net, '../weights/mobilenet_15.pth')
    
    # criterion
    criterion = nn.CrossEntropyLoss()

    # train processing
    test_model(net, num_classes, test_dataloader, criterion, device)
    
if __name__ == '__main__':
    test()