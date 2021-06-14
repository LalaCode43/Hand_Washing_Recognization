from lib import *

torch.manual_seed(4)
np.random.seed(4)
random.seed(4)
root_path = '/media/vas/54D6ABF9D6ABDA0E/Documents and Settings/data'
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 80
num_epochs = 35
num_classes = 7
# mean = [0.4294, 0.4446, 0.4264]
# std = [0.2134, 0.2001, 0.1820]

