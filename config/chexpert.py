############### Pytorch CHEXPERT SETTING 2 configuration file ###############
import math

start_epoch = 1
num_epochs = 200
batch_size = 16
optim_type = 'SGD'

# optimizer params
momentum = 0.9
weight_decay = 5e-4

lr_milestones = [60, 120, 160]
lr_gamma = 0.2

# network architecture
dropout = 0.3 
depth = 100
widen_factor = 2
lr = 0.01

# data parameters
validation_size = 0.1

image_size = 224

mean = {
    'chexpert': (0.5330,0.5330,0.5330)
}

std = {
    'chexpert': (0.0349,0.0349,0.0349)
}

classes_in = {
    'setting1': ['Cardiomegaly','Pneumothorax'],
    'setting2': ['Lung Opacity','Pleural Effusion']
}

classes_out = {
    'setting1': ['Fracture'],
    'setting2': ['Fracture', 'Pneumonia']
}

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
