from lib import *
from config import *
from utils import *
from imagetransform import *


class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        
        img_path = self.file_list[idx]
        # img = Image.open(img_path)
        img = cv2.imread(img_path)
        
        # change BGR -> RGB 
        img = img[:, :, (2, 1, 0)]

        # change to PIL image
        img = Image.fromarray(img)

        img_transformed = self.transform(img, self.phase)
        
        label = int(img_path.split(os.path.sep)[-2])
        return img_transformed, label


if __name__ == '__main__':
    trainvalset = make_data_path_list(phase='trainval')
    trainset, valset = data.random_split(trainvalset, (391409, 105890))
    
    # list of paths to images of train and val dataset
    train_list = [file for file in trainset]
    val_list = [file for file in valset]
    idx = 25
    img = cv2.imread(train_list[idx])
    cv2.imshow('test', img)
    cv2.waitKey(0)
    train_dataset = MyDataset(train_list, ImageTransform(resize, mean, std), phase='train')
    img_transformed, label = train_dataset[idx]
    print(img_transformed[:, 0:10, 0:10])
    # img_transformed = img_transformed.numpy().transpose(1, 2, 0)
    # img_transformed = np.clip(img_transformed, 0, 1)
    #
    # print(label)
    # cv2.imshow('test', img_transformed*255)
    # cv2.waitKey(0)