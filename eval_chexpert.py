# from __future__ import print_function

import argparse
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.autograd import Variable
from types import SimpleNamespace

from models.wide_resnet import Wide_ResNet
from util.metrics import get_metrics
from config import chexpert as cf
from dataloaders.chexpert import fetch_dataloader, selectClasses
from methods import mcp, odin, mcdp, deepensemble, mahalanobis, mahalanobis_ensemble

parser = argparse.ArgumentParser(description='PyTorch CheXpert Evaluation')

parser.add_argument('--setting', default='setting1', type=str,
                    help='dataset setting for CheXpert, either setting1 (default) or setting2')
parser.add_argument('--debug', '-d', action='store_true',
                    help='Debug Mode with 1 epoch')
parser.add_argument('--method', '-m', type=str, default='mcp',
                    help='Method for OOD Detection, one of [MCP (default), ODIN, MCDP, Mahalanobis, DeepEnsemble]')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
print('Using GPU: {}'.format(torch.cuda.device_count()))
best_acc = 0

dataset_name = 'chexpert'

batch_size = 5 if args.debug else cf.batch_size

resize = cf.image_size # image size

classes_in = cf.classes_in[args.setting]
classes_out = cf.classes_out[args.setting]

num_classes = len(classes_in)

# this loads the best model
model_id = str(cf.best)

# this should come from args in the future
ood_set = classes_out[0]
save_dir = 'outputs'


print('\n[Phase 1] : Data Preparation')

# transformations for both OOD and test since they come from the same distribution
transforms = T.Compose([
    T.Resize(resize),
    T.CenterCrop(resize),
    T.ToTensor(),
    T.Normalize(mean=[0.5330], std=[0.0349]),                                           # whiten with dataset mean and std
    lambda x: x.expand(3,-1,-1)])                                                       # expand to 3 channels

# load dataframe 
root = './datasets/CheXpert-v1.0-small'
path_valid = os.path.join(root, 'valid.csv')
data = pd.read_csv(path_valid)

print("| Preparing CheXpert test with the following classes: ")
print(f'| Classes in: {classes_in}')
print(f'| Classes out: {classes_out}')

data_modified = data.fillna(value=0)
# policy how to replace the values
policy = 0.0
print(f'All uncertain labels will be replaced with: {policy} (1 == positive, 0 == negative)')
data_modified = data_modified.replace(-1.0, policy)
# return only the needed classes 
test_dataset = selectClasses(data_modified, classes_in, classes_out)

# return only the ood class - we can simply switch this here
ood_dataset = selectClasses(data_modified, classes_out, classes_in)


print(f'| Length of OOD dataset: {len(ood_dataset)}')
print(f'| Length of TEST dataset: {len(test_dataset)}')

# get dataloaders
args_test = {'resize': resize, 'batch_size': batch_size, 'shuffle': False, 'root': 'datasets/'}
testloader = fetch_dataloader(args=SimpleNamespace(**args_test), dataframe=test_dataset, transforms=transforms)

oodloader = fetch_dataloader(args=SimpleNamespace(**args_test), dataframe=ood_dataset, transforms=transforms)

# do the same for the training loader for Mahalanobis
path_train = os.path.join(root, 'train.csv')
data = pd.read_csv(path_train)

data_modified = data.fillna(value=0)
# policy how to replace the values
policy = 0.0
print(f'All uncertain labels will be replaced with: {policy} (1 == positive, 0 == negative)')
data_modified = data_modified.replace(-1.0, policy)
# return only the needed classes 
dataset = selectClasses(data_modified, classes_in, classes_out)

# split into train/val/test 
if args.debug:
    df_train = dataset.sample(n=10,random_state=42)
    df_validation = dataset.drop(df_train.index)
    df_validation = df_validation.sample(n=10, random_state=42)
else:
    df_validation = dataset.sample(frac=0.1,random_state=42)
    df_train = dataset.drop(df_validation.index)

print(f'| Length of train dataset: {len(df_train)}')
print(f'| Length of validation dataset: {len(df_validation)}')

# get dataloaders
args_train = {'resize': resize, 'batch_size': batch_size, 'shuffle': True, 'root': 'datasets/'}
trainloader = fetch_dataloader(args=SimpleNamespace(**args_train), dataframe=df_train, transforms=transforms)

# Return network & file name
net = Wide_ResNet(depth, widen_factor, dropout, num_classes)
file_name = f'wide-resnet-{depth}x{widen_factor}_cheXpert-{model_id}'

print('| Loading model: {}'.format(file_name))

# Test only option
print('\n[Test Phase] : Model setup')
assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
checkpoint = torch.load('./checkpoint/'+dataset_name+os.sep+file_name+'.pth')
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
    extensions = ['527978', '539077', '64']
    nets = []
    for e in extensions:
        net = Wide_ResNet(depth, widen_factor,
                          dropout, num_classes)
        file_name = f'wide-resnet-{depth}x{widen_factor}_cheXpert-{e}'
        checkpoint = torch.load(
            './checkpoint/'+dataset_name+os.sep+file_name+'.pth')
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
              testloader=testloader, oodloader=oodloader, magnitude=0.0, num_classes=len(classes_in), save_dir=save_dir)
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))
    auroc, aucpr, fpr, tpr = get_metrics(f1_path, f2_path)
    print('AUROC for {}: {}\nAUCPR: {}'.format(method, auroc, aucpr))
    return auroc


if args.method.lower() == 'odin':
    eval_func = odin.eval
    method = 'ODIN'
    ood_loop(eval_func, ood_set, method, save_dir, net, testloader, oodloader)
    # tempering only evaluation
    method = 'ODIN_TEMPERING_ONLY'
    eval_func = odin_t.eval
    ood_loop(eval_func, ood_set, method, save_dir, net, testloader, oodloader)
    # adversarial only evaluation
    method = 'ODIN_ADVERSARIAL_ONLY'
    eval_func = odin_adv.eval
    ood_loop(eval_func, ood_set, method, save_dir, net, testloader, oodloader)
    sys.exit(0)
elif args.method.lower() == 'mcdp':
    eval_func = mcdp.eval
    method = 'MCDP'
    # net = load_mcdp_net()
elif args.method.lower() == 'deepensemble' or args.method.lower() == 'de':
    eval_func = deepensemble.eval
    method = 'Ensemble'
    net = load_nets()  # watch out, this actually loads multiple nets
elif args.method.lower() == 'mcdpensemble' or args.method.lower() == 'mcen':
    eval_func = mcdp_ensemble.eval
    method = 'MonteCarloEnsemble'
    net = load_nets()  # watch out, this actually loads multiple nets
elif args.method.lower() == 'mahalanobis' or args.method.lower() == 'ma':
    method = 'Mahalanobis'
    ood_loop_mahalanobis(mahalanobis.eval, ood_set, method,
                         save_dir, net, testloader, oodloader, trainloader)
    sys.exit(0)
elif args.method.lower() == 'mahalanobis-ensemble' or args.method.lower() == 'me':
    eval_func = mahalanobis_ensemble.eval
    method = 'MahalanobisEnsemble'
    nets = load_nets()  # watch out, this actually loads multiple nets
    ood_loop_mahalanobis(mahalanobis_ensemble.eval, ood_set, method,
                         save_dir, nets, testloader, oodloader, trainloader)
    sys.exit(0)
else:
    eval_func = mcp.eval
    method = 'MCP'


if args.method.lower() == 'all_ensembles':
    t0 = time.time()
    # load deep ensemble nets
    nets = load_nets()
    ensemble_auroc = ood_loop(
        deepensemble.eval, ood_set, 'DeepEnsemble', save_dir, nets, testloader, oodloader)
    mahal_ensemble_auroc = ood_loop_mahalanobis(mahalanobis_ensemble.eval, ood_set, 'MahalanobisEnsemble',
                                                save_dir, nets, testloader, oodloader, trainloader)
    print(
        f'AUROC\nMahalanobis Ensemble: {mahal_ensemble_auroc}\nDeep Ensemble: {ensemble_auroc}')
    print(f'Elapsed time: {t0 - time.time()}')
elif args.method.lower() == 'all_single':
    t0 = time.time()
    mcp_auroc = ood_loop(mcp.eval, ood_set, 'MCP',
                         save_dir, net, testloader, oodloader)
    odin_auroc = ood_loop(odin.eval, ood_set, 'ODIN',
                          save_dir, net, testloader, oodloader)
    # tempering only evaluation
    method = 'ODIN_TEMPERING_ONLY'
    odin_t_auroc = ood_loop(odin_t.eval, ood_set, method, save_dir, net, testloader, oodloader)
    # adversarial only evaluation
    method = 'ODIN_ADVERSARIAL_ONLY'
    odin_adv_auroc = ood_loop(odin_adv.eval, ood_set, method, save_dir, net, testloader, oodloader)
    #net = load_mcdp_net()
    mcdp_auroc = ood_loop(mcdp.eval, ood_set, 'MCDP',
                          save_dir, net, testloader, oodloader)
    mahal_auroc = ood_loop_mahalanobis(mahalanobis.eval, ood_set, 'Mahalanobis',
                                       save_dir, net, testloader, oodloader, trainloader)
    print(
        f'AUROC\nMCP:  {mcp_auroc}\nODIN: {odin_auroc}\nODIN ADV: {odin_adv_auroc}\nODIN TEMP: {odin_t_auroc}\nMCDP: {mcdp_auroc}\nMahalanobis: {mahal_auroc}')
    print(f'Elapsed time: {t0 - time.time()}')
else:
    ood_loop(eval_func, ood_set, method, save_dir, net, testloader, oodloader)
