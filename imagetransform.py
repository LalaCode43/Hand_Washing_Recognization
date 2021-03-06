from lib import *
from config import *


class ImageTransform:
    def __init__(self, resize, mean, std):
        # self.data_transform = {
        #     'train': transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor()
        #     ]),
        #     'val': transforms.Compose([
        #         transforms.Resize((256, 256)),
        #         transforms.ToTensor()
        #     ])
        # }

        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(256, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0, hue=0),
                transforms.RandomRotation(degrees=20),
                transforms.CenterCrop(size=resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(size=resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        }
        # self.data_transform = {
        #     'train': transforms.Compose([
        #         transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        #         transforms.RandomRotation(degrees=20),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.CenterCrop(size=resize),
        #         transforms.ToTensor()
        #     ]),
        #     'val': transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(size=resize),
        #         transforms.ToTensor()
        #     ])
        # }
    
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


if __name__ == '__main__':
    trans = ImageTransform(resize, mean, std)
    img = cv2.imread('dog.jpg')

    img = Image.fromarray(img)   
    
    img_transformed = trans(img, phase='train')
    img_transformed = img_transformed.numpy().transpose(1, 2, 0)
    img_transformed = np.clip(img_transformed, 0, 1)
    cv2.imshow('test', img_transformed)
    cv2.waitKey(0)