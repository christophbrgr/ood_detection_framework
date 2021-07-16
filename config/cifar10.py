############### Pytorch CIFAR configuration file ###############
## This and the files for wide resnet are modified from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/config.py
import math


start_epoch = 1
num_epochs = 200
batch_size = 128
optim_type = 'SGD'

# optimizer params
momentum = 0.9
weight_decay = 5e-4

lr_milestones = [60, 120, 160]
lr_gamma = 0.2

# network architecture
dropout = 0.3 
depth = 28
widen_factor = 10
lr = 0.1

# data parameters
validation_size = 0.1

image_size = 32

# values for svhn from: https://deepobs.readthedocs.io/en/develop/_modules/deepobs/pytorch/datasets/svhn.html
mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'svhn': (0.4376821, 0.4437697, 0.47280442),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'svhn': (0.19803012, 0.20101562, 0.19703614),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
