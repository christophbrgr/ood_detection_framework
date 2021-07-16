from __future__ import print_function

import argparse
import datetime
import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np

import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from config import chexpert as cf
from models.wide_resnet import Wide_ResNet
from dataloaders.chexpert import fetch_dataloader, selectClasses
import wandb

parser = argparse.ArgumentParser(description='PyTorch CheXpert Training')
parser.add_argument('--setting', default='setting1', type=str,
                    help='dataset setting for CheXpert, either setting1 (default) or setting2')
parser.add_argument('--wandb', '-w', action='store_true', type=bool, default=True, 
                    help='Weights and Biases Logging (requires login)')
parser.add_argument('--debug', '-d', action='store_true', type=bool, default=False, 
                    help='Debug Mode with 1 epoch')
parser.add_argument('--id', '-i', type=int, default=0,
                    help='ID of the run')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
print('Using GPU: {}'.format(torch.cuda.device_count()))
best_acc = 0

dataset_name = 'chexpert'

start_epoch = cf.start_epoch
num_epochs = cf.num_epochs
batch_size = 1 if args.debug else cf.batch_size
optim_type = cf.optim_type

resize = cf.image_size # image size

classes_in = cf.classes_in[args.setting]
classes_out = cf.classes_out[args.setting]

num_classes = len(classes_in)

if args.debug:
    num_epochs = 50
    print('DEBUG MODE WITH {num_epochs} EPOCHS ONLY')

# setup checkpoint and experiment tracking
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
save_point = './checkpoint/'+dataset_name+os.sep
if not os.path.isdir(save_point):
    os.mkdir(save_point)
experiment_runs = len(os.listdir(save_point))
print('| Number of experiments saved: {}'.format(experiment_runs))


#### Get a random ID #####
random.seed()
if args.id == 0:
    experiment_run = random.randint(experiment_runs+1, 1000000)
else:
    experiment_run = args.id
print('| ID of this run: {}'.format(experiment_run))


# load dataframe 
root = './datasets/CheXpert-v1.0-small'
path_train = os.path.join(root, 'train.csv')
data = pd.read_csv(path_train)

print("| Preparing CheXpert dataset with the following classes: ")
print(f'| Classes in: {classes_in}')
print(f'| Classes out: {classes_out}')

data_modified = data.fillna(value=0)
# policy how to replace the values
policy = 0.0
print(f'All uncertain labels will be replaced with: {policy} (1 == positive, 0 == negative)')
data_modified = data_modified.replace(-1.0, policy)
# return only the needed classes 
dataset, weights = selectClasses(data_modified, classes_in, classes_out)

# split into train/val/test 
if args.debug:
    df_train = dataset.sample(n=10,random_state=42)
    df_validation = dataset.drop(df_train.index)
    df_validation = df_validation.sample(n=10, random_state=42)
else:
    df_validation = dataset.sample(frac=0.2,random_state=42)
    df_train = dataset.drop(df_validation.index)
    df_test = df_validation.sample(frac=0.5, random_state=42)
    df_validation = df_validation.drop(df_test.index)
    df_test.to_csv(os.path.join(root, f'test_{experiment_run}.csv'))

print(f'| Length of train dataset: {len(df_train)}')
print(f'| Length of validation dataset: {len(df_validation)}')
print(f'| Length of test dataset: {len(df_test)}')
print(f'| Weights to balance the classes: {weights}')

# get dataloaders
args_train = {'resize': resize, 'batch_size': batch_size, 'shuffle': True, 'root': 'datasets/'}
trainloader = fetch_dataloader(args=SimpleNamespace(**args_train), dataframe=df_train, weights=weights)

args_validation = {'resize': resize, 'batch_size': batch_size, 'shuffle': False, 'root': 'datasets/'}
validationloader = fetch_dataloader(args=SimpleNamespace(**args_validation), dataframe=df_validation)

# Return network & file name
def getNetwork(args):
    net = Wide_ResNet(cf.depth, cf.widen_factor,
                        cf.dropout, num_classes)
    file_name = 'wide-resnet-'+str(cf.depth)+'x'+str(cf.widen_factor)+'_cheXpert'
    return net, file_name


# Model
print('\n[Phase 2] : Model setup')
print('| Building net type [wideresnet]...')
net, file_name = getNetwork(args)
# net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

# we only train on exclusive classes!!
criterion = nn.CrossEntropyLoss()

# optimizer params
momentum = cf.momentum
weight_decay = cf.weight_decay

if args.wandb:
    # Set up logging and pass params to wandb
    wandb.init(project="master-thesis", entity="cberger",
            name=f'wrn_{cf.depth}x{cf.widen_factor}_SGD_{dataset_name}_{args.setting}_{len(classes_in)}in{len(classes_out)}out_{experiment_run}', config=args)
    wandb.config.batch_size = batch_size
    wandb.config.file_name = file_name
    wandb.config.optim_momentum = momentum
    wandb.config.optim_weight_decay = weight_decay
    wandb.config.image_size = resize
    wandb.config.classes_in = classes_in
    wandb.config.classes_out = classes_out

    wandb.watch(net)

optimizer = torch.optim.SGD(net.parameters(), lr=cf.lr, momentum=cf.momentum, weight_decay=cf.weight_decay)
lr_milestones = cf.lr_milestones
lr_gamma = cf.lr_gamma

if args.wandb:
    wandb.config.lr_milestones = lr_milestones
    wandb.config.lr_gamma = lr_gamma

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=lr_milestones, gamma=lr_gamma
)

def train(epoch):
    np.random.seed() # reset seed to ensure that our data split is actually random
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0

    print('\n=> Training Epoch #%d, LR=%.4f' %
          (epoch, scheduler.get_last_lr()[0]))
    if args.wandb:
        wandb.log({"epoch": epoch})
        wandb.log({"lr": scheduler.get_last_lr()[0]})

    for batch_idx, (inputs, targets, pat_id) in enumerate(trainloader):
        #print(f'labels: {targets}')
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        if args.debug:
            print(f'Inputs shape: {inputs.shape}')
            print(f'Output shape: {outputs.shape}')
            print(f'Targets shape: {targets.shape}')
            print(f'Targets: {targets}')
            print(f'Outputs: {outputs}')
        # assert outputs.shape == targets.shape, 'Outputs do not match targets!'
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        accuracy = 100.*correct/total
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, batch_idx+1,
                            (len(trainloader.dataset)//batch_size)+1, loss.item(), accuracy))
        if args.wandb:
            wandb.log({"train_loss": loss.item()})
            wandb.log({"train_acc": accuracy})
        sys.stdout.flush()
    scheduler.step()

def validate(epoch):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, pat_id) in enumerate(validationloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %
              (epoch, loss.item(), acc))

        if args.wandb:
            wandb.log({"val_loss": loss.item()})
            wandb.log({"val_acc": acc})

        if not args.debug and acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
            # save the state with the number of the run
            torch.save(net.state_dict(), save_point +
                       file_name+'-'+str(experiment_run)+'.pth')
            best_acc = acc


print('\nTraining model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    validate(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

print('* Validation results : Acc@1 = %.2f%%' % (best_acc))
with open((save_point+file_name+'-'+str(experiment_run)+'.txt'), 'w') as f:
    f.write('Run: {}\nValidation Accuracy: {}\nDataset: {}'.format(
        experiment_run, best_acc, dataset_name))
print('| Saved all results to file. Training done.')
