from lib import *
from utils import *
from imagetransform import *
from mydataset import *
from config import *

import argparse


def parse_inputs():
    parser = argparse.ArgumentParser(description='Hand Washing Recognization')
    
    parser.add_argument(
        '--file_dir',
        metavar='F',
        type=str,
        default='../imgs/1_press_set1_pass_dwn_frame011.jpg'
    )
    
    parser.add_argument(
        '--weight_dir',
        metavar='W',
        type=str,
        default='weights/mobilenet_29.pth'
    )
    
    parser.add_argument(
        '--model_name',
        metavar='M',
        type=str,
        default='mobilenet'
    )

    parser.add_argument(
        '--thresh_hold',
        metavar='T',
        type=float,
        default=0.75
    )
    
    args = parser.parse_args()
    
    return args

def initialize_model(model_name, num_classes, use_pretrained = True):
    if model_name == 'mobilenet':
         # load pretrained model MobileNet_V2
        net = models.mobilenet_v2(pretrained=use_pretrained)
        num_ftrs = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(num_ftrs, num_classes)
        return net
    

def main(args):
    model_name = args.model_name 
    file_dir = args.file_dir
    threshold = args.thresh_hold
    weight_dir = args.weight_dir
    
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # initialize_model
    net = initialize_model(model_name, 6, False)
    net.to(device)
    
    # load mode weight
    net = load_model(net, weight_dir)
    
    # set model to val phase
    net.eval()
    
    # softmax function 
    sm = nn.Softmax(dim=1)

    if file_dir.split('.')[-1] == 'jpg': # if input is image
        # read image
        img = cv2.imread(file_dir)
        
        transform = ImageTransform(resize, mean, std)
        
        
        img_transformed = transform(img)
        img_transformed = img_transformed.unsqueeze_(0)
        

        output = sm(net(img_transformed))
        print(output)
        ret, preds = torch.max(output, dim=1)
        print(preds.item() + 1)
        cv2.imshow('img', img)
        cv2.waitKey(0)        
        
        

if __name__ == '__main__':
    args = parse_inputs()
    main(args)