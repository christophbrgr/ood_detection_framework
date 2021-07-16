# from __future__ import print_function

import argparse
import datetime
import os
import sys
import time

import numpy as np
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
from util.metrics import get_metrics
from methods import mcp, odin, mcdp, deepensemble, mahalanobis, mahalanobis_ensemble, odin_t, odin_adv

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--debug', '-d', action='store_true',
                    help='Debug Mode with 1 epoch')
parser.add_argument('--method', '-m', type=str,
                    help='Method for OOD Detection, one of [MCP (default), ODIN, MCDP, Mahalanobis, DeepEnsemble, MahalanobisEnsemble, all]')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
print('Using GPU: {}'.format(torch.cuda.device_count()))
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

depth = cf.depth
widen_factor = cf.widen_factor
dropout = cf.dropout
lr = cf.lr 

# this should come from args in the future
ood_set = 'svhn'
dataset_name = 'cifar10'
save_dir = os.path.join('outputs', f'run_cifar10')

batch_size = 128

print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

transform_ood = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[ood_set], cf.std[ood_set]),
])

print("| Preparing CIFAR-10 dataset...")
sys.stdout.write("| ")
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
num_classes = 10

# prep ood dataset
if ood_set == 'svhn':
    print("| Preparing SVHN dataset for OOD detection...")
    sys.stdout.write("| ")
    oodset = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform_ood)
    oodloader = torch.utils.data.DataLoader(
        oodset, batch_size=batch_size, shuffle=False, num_workers=4)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=4)
trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Return network & file name
net = Wide_ResNet(depth, widen_factor, dropout, num_classes)
file_name = 'wide-resnet-'+str(depth)+'x'+str(widen_factor)+'-17'

print('| Loading model: {}'.format(file_name))

# Test only option
print('\n[Test Phase] : Model setup')
assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.pth')
# print(checkpoint.keys())
# adapt net to state
params = {}
for k_old in checkpoint.keys():
    k_new = k_old.replace('module.', '')
    params[k_new] = checkpoint[k_old]
net.load_state_dict(params)

if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(
    #     net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


def load_nets():
    # for deep ensembles
    extensions = ['-17', '-18', '-15', '-24', '-23']
    nets = []
    for e in extensions:
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)+e
        checkpoint = torch.load(
            './checkpoint/'+args.dataset+os.sep+file_name+'.t7')
        net = checkpoint['net']
        if use_cuda:
            net.cuda()
            net = torch.nn.DataParallel(
                net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
        nets.append(net)
    return nets


def load_nets_mahalanobis():
    # for deep ensembles
    extensions = ['-17', '-18', '-15', '-24', '-23']
    nets = []
    for e in extensions:
        net = Wide_ResNet(args.depth, args.widen_factor,
                          args.dropout, num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)+e
        checkpoint = torch.load(
            './checkpoint/'+args.dataset+os.sep+file_name+'.pth')
        # adapt net to state
        params = {}
        for k_old in checkpoint.keys():
            k_new = k_old.replace('module.', '')
            params[k_new] = checkpoint[k_old]
        net.load_state_dict(params)

        if use_cuda:
            net.cuda()
            cudnn.benchmark = True
        nets.append(net)
    return nets


def ood_loop(eval_func, ood_set, method, save_dir, net, testloader, oodloader):
    print('\n| [OOD] : Testing OOD detection with {} using {}'.format(
        ood_set, method))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f1_path = os.path.join(save_dir, "confidence_IN_{}.txt".format(method))
    f2_path = os.path.join(save_dir, "confidence_OUT_{}.txt".format(method))
    print('\n| Testing with {}'.format(method))
    elapsed_time = 0
    start_time = time.time()
    # testing OOD dataset
    eval_func(f1_path, f2_path, net, testloader, oodloader, save_dir=save_dir)
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))
    auroc, aucpr, fpr, tpr = get_metrics(f1_path, f2_path)
    print('AUROC for {}: {}\nAUCPR: {}'.format(method, auroc, aucpr))
    return auroc


def ood_loop_mahalanobis(eval_func, ood_set, method, save_dir, net, testloader, oodloader, trainloader):
    print('\n| [OOD] : Testing OOD detection with {} using {}'.format(
        ood_set, method))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f1_path = os.path.join(save_dir, "confidence_IN_{}.txt".format(method))
    f2_path = os.path.join(save_dir, "confidence_OUT_{}.txt".format(method))
    print('\n| Testing with {}'.format(method))
    elapsed_time = 0
    start_time = time.time()
    # testing OOD dataset
    eval_func(f1_path, f2_path, net, trainloader=trainloader,
              testloader=testloader, oodloader=oodloader, magnitude=0.0)
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))
    auroc, aucpr, fpr, tpr = get_metrics(f1_path, f2_path)
    print('AUROC for {}: {}\nAUCPR: {}'.format(method, auroc, aucpr))
    return auroc


if args.method.lower() == 'odin':
    eval_func = odin.eval_cifar10
    method = 'ODIN'
    ood_loop(eval_func, ood_set, method, save_dir, net, testloader, oodloader)
    # tempering only evaluation
    method = 'ODIN_TEMPERING_ONLY'
    eval_func = odin_t.eval_cifar10
    ood_loop(eval_func, ood_set, method, save_dir, net, testloader, oodloader)
    # adversarial only evaluation
    method = 'ODIN_ADVERSARIAL_ONLY'
    eval_func = odin_adv.eval_cifar10
    ood_loop(eval_func, ood_set, method, save_dir, net, testloader, oodloader)
    sys.exit(0)
elif args.method.lower() == 'mcdp':
    eval_func = mcdp.eval_cifar10
    method = 'MCDP'
elif args.method.lower() == 'deepensemble' or args.method.lower() == 'de':
    eval_func = deepensemble.eval_cifar10
    method = 'Ensemble'
    net = load_nets()  # watch out, this actually loads multiple nets
elif args.method.lower() == 'mahalanobis' or args.method.lower() == 'ma':
    method = 'Mahalanobis'
    ood_loop_mahalanobis(mahalanobis.eval_cifar10, ood_set, method,
                         save_dir, net, testloader, oodloader, trainloader)
    sys.exit(0)
elif args.method.lower() == 'mahalanobis-ensemble' or args.method.lower() == 'me':
    eval_func = mahalanobis_ensemble.eval
    method = 'MahalanobisEnsemble'
    nets = load_nets_mahalanobis()  # watch out, this actually loads multiple nets
    ood_loop_mahalanobis(mahalanobis_ensemble.eval_cifar10, ood_set, method,
                         save_dir, nets, testloader, oodloader, trainloader)
    sys.exit(0)
else:
    eval_func = mcp.eval_cifar10
    method = 'MCP'


if args.method.lower() == 'all':
    t0 = time.time()
    mcp_auroc = ood_loop(mcp.eval_cifar10, ood_set, 'MCP',
                         save_dir, net, testloader, oodloader)
    odin_auroc = ood_loop(odin.eval_cifar10, ood_set, 'ODIN',
                          save_dir, net, testloader, oodloader)
    mcdp_auroc = ood_loop(mcdp.eval_cifar10, ood_set, 'MCDP',
                          save_dir, net, testloader, oodloader)
    mahal_auroc = ood_loop_mahalanobis(mahalanobis.eval_cifar10, ood_set, 'Mahalanobis',
                                       save_dir, net, testloader, oodloader, trainloader)
    # load deep ensemble nets
    nets = load_nets_mahalanobis()
    ensemble_auroc = ood_loop(
        deepensemble.eval_cifar10, ood_set, 'DeepEnsemble', save_dir, nets, testloader, oodloader)
    mahal_ensemble_auroc = ood_loop_mahalanobis(mahalanobis_ensemble.eval_cifar10, ood_set, 'MahalanobisEnsemble',
                                                save_dir, nets, testloader, oodloader, trainloader)
    print(
        f'AUROC\nMCP:  {mcp_auroc}\nODIN: {odin_auroc}\nMCDP: {mcdp_auroc}\nMahalanobis: {mahal_auroc}\nMahalanobis Ensemble: {mahal_ensemble_auroc}\nDeep Ensemble: {ensemble_auroc}')
    print(f'Elapsed time: {t0 - time.time()}')

else:
    ood_loop(eval_func, ood_set, method, save_dir, net, testloader, oodloader)
