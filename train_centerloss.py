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

from util.centerloss import CenterLoss

import config as cf
from models.wide_resnet_centerloss import Wide_ResNet
import wandb

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet',
                    type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10,
                    type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--centroid_size', '-c', default=512, type=int, help='centroid size')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--valSize', '-v', default=0.1, type=float,
                    help='Specify the size of the validation set as a fraction of the training set. Default: 0.1')
parser.add_argument('--testOnly', '-t', action='store_true',
                    help='Test mode with the saved model')
parser.add_argument('--ensemble', '-e', action='store_true',
                    help='Deep ensemble mode to choose a new seed')
parser.add_argument('--debug', '-d', action='store_true',
                    help='Debug Mode with 1 epoch')
args = parser.parse_args()

centroid_size = args.centroid_size

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
print('Using GPU: {}'.format(torch.cuda.device_count()))
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

if args.debug:
    print('DEBUG MODE WITH 1 EPOCH ONLY')
    num_epochs = 1

# setup checkpoint and experiment tracking
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
save_point = './checkpoint/'+args.dataset+os.sep
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
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])  # meanstd transformation

transform_validation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
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
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

validationloader = torch.utils.data.DataLoader(
    validationset, batch_size=batch_size, shuffle=False, num_workers=2)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Return network & file name


def getNetwork(args):
    if (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor,
                          args.dropout, num_classes, feature_dim=centroid_size)
        file_name = f'wide-resnet-{args.depth}x{args.widen_factor}-centerloss'
    else:
        print(
            'Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name


# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load(
        './checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    # net.apply(conv_init)

if use_cuda:
    net.cuda()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# what's the feature dim? Hidden layer with x number of features used as our centroid before the final classification layer
center_loss = CenterLoss(num_classes=num_classes, feat_dim=centroid_size, use_gpu=True)
lr_cent = 0.5
alpha = 0.3

# optimizer params
momentum = 0.9
weight_decay = 5e-4

# Set up logging and pass params to wandb
wandb.init(project="master-thesis", entity="cberger",
           name=f'wrn_centerloss_a{alpha}_c{centroid_size}_{experiment_run}', config=args)
wandb.config.batch_size = batch_size
wandb.config.file_name = file_name
wandb.config.center_loss_lr = lr_cent
wandb.config.center_loss_alpha = alpha
wandb.config.optim_momentum = momentum
wandb.config.optim_weight_decay = weight_decay
wandb.config.centroid_size = centroid_size

wandb.watch(net)


def train(epoch):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    lr=cf.learning_rate(args.lr, epoch)
    # joint optimizer for both losses
    params = list(net.parameters()) + list(center_loss.parameters())
    optimizer = optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay)

    print('\n=> Training Epoch #%d, LR=%.4f' %
          (epoch, cf.learning_rate(args.lr, epoch)))
    wandb.log({"epoch": epoch})
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs, features = net.penultimate_forward(inputs)               # Forward Propagation
        # print(f'Feature shape: {features.shape}')

        cl = center_loss(features, targets)

        loss = cl * alpha + criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        for param in center_loss.parameters():
            # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
            param.grad.data *= (lr_cent / (alpha * lr))
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        accuracy = 100.*correct/total
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss.item(), accuracy))
        wandb.log({"train_loss": loss.item()})
        wandb.log({"train_total_loss": train_loss})
        wandb.log({"train_acc": accuracy})
        wandb.log({"train_centerloss": cl.item()})
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

        wandb.log({"val_loss": loss.item()})
        wandb.log({"val_acc": acc})

        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
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
        experiment_run, best_acc, args.dataset))
print('| Saved all results to file. Training done.')
