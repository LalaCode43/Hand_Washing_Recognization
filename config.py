from lib import *

torch.manual_seed(4)
np.random.seed(4)
random.seed(4)
root_path = '/media/vas/54D6ABF9D6ABDA0E/Documents and Settings/data'
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 64
num_epochs = 20
num_classes = 7

# mean  = [0.4172, 0.4346, 0.4198]
# std = [0.1786, 0.1957, 0.2089]
# mean = [0.4653, 0.4865, 0.4727]
# std = [0.1594, 0.1766, 0.1901]

