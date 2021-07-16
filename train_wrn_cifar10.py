from __future__ import print_function

import argparse
import datetime
import os
import sys
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from config import cifar10 as cf
from models.wide_resnet import Wide_ResNet
import wandb

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--wandb', '-w', action='store_true', type=bool, default=True, 
                    help='Weights and Biases Logging (requires login)')
parser.add_argument('--debug', '-d', action='store_true', type=bool, default=False, 
                    help='Debug Mode with 1 epoch')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
print('Using GPU: {}'.format(torch.cuda.device_count()))
best_acc = 0

dataset_name = 'cifar10'

start_epoch = cf.start_epoch
num_epochs = cf.num_epochs
batch_size = 1 if args.debug else cf.batch_size
optim_type = cf.optim_type

print_debug = 0

if args.debug:
    print('DEBUG MODE WITH 1 EPOCH ONLY')
    num_epochs = 1
    batch_size = 1
    print_debug = True

# setup checkpoint and experiment tracking
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
save_point = './checkpoint/'+dataset_name+os.sep
if not os.path.isdir(save_point):
    os.mkdir(save_point)
experiment_runs = len(os.listdir(save_point))
print('| Number of experiments saved: {}'.format(experiment_runs))
experiment_run = experiment_runs + 1
print('| ID of this run: {}'.format(experiment_run))

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[dataset_name], cf.std[dataset_name]),
])  # meanstd transformation

transform_validation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[dataset_name], cf.std[dataset_name]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[dataset_name], cf.std[dataset_name]),
])

print("| Preparing CIFAR-10 dataset...")
sys.stdout.write("| ")
dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=None)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
validation_length = int(len(dataset)*args.valSize)
train_length = len(dataset) - validation_length
print('Size of Validation Set: {}\nSize of Training Set: {}'.format(
    validation_length, train_length))
trainset, validationset = torch.utils.data.random_split(
    dataset, [train_length, validation_length])
trainset.dataset.transform = transform_train
validationset.dataset.transform = transform_validation
num_classes = 10


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

validationloader = torch.utils.data.DataLoader(
    validationset, batch_size=batch_size, shuffle=False, num_workers=2)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Return network & file name
def getNetwork(args):
    net = Wide_ResNet(cf.depth, cf.widen_factor,
                        cf.dropout, num_classes)
    file_name = 'wide-resnet-'+str(cf.depth)+'x'+str(cf.widen_factor)
    return net, file_name


# Model
print('\nModel setup')
print('| Building net type [wideresnet]...')
net, file_name = getNetwork(args)
# net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Set up logging and pass params to wandb
momentum = cf.momentum
weight_decay = cf.weight_decay

if args.wandb:
    # Set up logging and pass params to wandb
    wandb.init(project="master-thesis", entity="cberger",
            name=f'wrn_{args.depth}x{args.widen_factor}_vanilla_{args.dataset}_{experiment_run}', config=args)
    wandb.config.batch_size = batch_size
    wandb.config.file_name = file_name
    wandb.config.optim_momentum = momentum
    wandb.config.optim_weight_decay = weight_decay

    wandb.watch(net)

def train(epoch):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(
        cf.lr, epoch), momentum=momentum, weight_decay=weight_decay)

    print('\n=> Training Epoch #%d, LR=%.4f' %
          (epoch, cf.learning_rate(cf.lr, epoch)))
    if args.wandb:
        wandb.log({"epoch": epoch})
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if print_debug: 
            print(f'Labels shape: {targets.shape}')
            print(f'Labels: {targets[0]}')
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        if args.debug:
            print(f'Inputs shape: {inputs.shape}')
            print(f'Output shape: {outputs.shape}')
            print(f'Targets shape: {targets.shape}')
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
                            (len(trainset)//batch_size)+1, loss.item(), accuracy))
        if args.wandb:
            wandb.log({"train_loss": loss.item()})
            wandb.log({"train_total_loss": train_loss})
            wandb.log({"train_acc": accuracy})
        sys.stdout.flush()


def validate(epoch):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validationloader):
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

        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
            state = {
                'net': net.module if use_cuda else net,
                'acc': acc,
                'epoch': epoch,
            }
            # save the state with the number of the run
            torch.save(state, save_point+file_name +
                       '-'+str(experiment_run)+'.t7')
            torch.save(net.state_dict(), save_point +
                       file_name+'-'+str(experiment_run)+'.pth')
            best_acc = acc


print('\n[Phase 3] : Training model')
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
