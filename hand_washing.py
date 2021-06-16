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
        default='./result/14_6_2021/weights/mobilenet_28.pth'
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
    net = initialize_model(model_name, 7, False)
    net.to(device)
    
    # load mode weight
    net = load_model(net, weight_dir)
    
    # set model to val phase
    net.eval()
    
    # softmax function 
    sm = nn.Softmax(dim=1)
    
    # transform image
    transform = ImageTransform(resize, mean, std)
    
    

    if file_dir.split('.')[-1] == 'jpg': # if input is image file
        # read image
        img = cv2.imread(file_dir)
        frame = img[:, :, (2, 1, 0)]
        frame = Image.fromarray(frame)
    
        
        
        
        img_transformed = transform(frame, 'val')
        print(img_transformed.shape)
        img1 = img_transformed.numpy().transpose(1, 2, 0)
        img1 = np.clip(img1, 0, 1)
        img_transformed = img_transformed.unsqueeze_(0)
        img_transformed = img_transformed.to(device)
    
        output = sm(net(img_transformed))
        print(output)
        ret, preds = torch.max(output, dim=1)
        print(preds.item())
        cv2.imshow('img', img)
        cv2.waitKey(0)        
        cv2.imshow('img', img1)
        cv2.waitKey(0)    
    if file_dir.split('.')[-1] in ['mp4', 'avi']: # if input is video file
        vidcap = cv2.VideoCapture(file_dir)
        class_total = np.zeros(7, dtype=int)
        width = int(vidcap.get(3))
        height = int(vidcap.get(4))
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        FILE_OUTPUT = '../output.avi'
        out = cv2.VideoWriter(FILE_OUTPUT, fourcc, 30, (int(width), int(height)))
        while True:
            
            ret, frame_cv = vidcap.read() 
        
            if ret:
                frame = frame_cv[:, :, (2, 1, 0)]
                img = Image.fromarray(frame)
                
                input_frame  =  transform(img, 'val')
                
                input_frame = input_frame.to(device)
                
                input_frame = input_frame.unsqueeze_(0)
                
                output_frame = net(input_frame)          
                prob_outs = sm(output_frame)
                
                value, index = torch.max(prob_outs, dim=1)
                
                pvalue = value.cpu().item()
                pclass = index.cpu().item()
                class_total[pclass] += 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame_cv,
                                    str(pclass),
                                    (50, 50),
                                    font, 1,
                                    (0, 255, 255),
                                    2,
                                    cv2.LINE_4)
                cv2.imshow('my webcam', frame_cv)
                out.write(frame_cv)

                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                break
        vidcap.release()
        result.release()
        cv2.destroyAllWindows()
        print("The video was successfully saved")
        print('Total of step: ', class_total)
    
                
if __name__ == '__main__':
    args = parse_inputs()
    main(args)