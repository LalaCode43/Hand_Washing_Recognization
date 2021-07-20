from lib import *
from utils import *
from imagetransform import *
from mydataset import *
from config import *


trainvalset = make_data_path_list(phase='trainval')
trainset, valset = data.random_split(trainvalset, (568183, 164328),generator=torch.Generator().manual_seed(4))

# list of paths to images of train and val dataset
train_list = [file for file in trainset]
val_list = [file for file in valset]

train_dataset = MyDataset(file_list=train_list, transform=ImageTransform(1, 1, 1), phase='train')
val_dataset = MyDataset(file_list=val_list, transform=ImageTransform(1, 1, 1), phase='val')


train_dataloader = data.DataLoader(train_dataset,  2000, num_workers=1, shuffle=False)
val_dataloader = data.DataLoader(val_dataset,  128, num_workers=1, shuffle=False)

dataloader_dict = {
        'train': train_dataloader,
        'val': val_dataloader
        }

mean = 0.
std = 0.
nb_samples = 0.
for phase in ['train']:
    for data, label in tqdm(dataloader_dict[phase]):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
print(mean)
print(std)