"""
Based on: https://github.com/y0ast/deterministic-uncertainty-quantification
"""
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import os
import sys
import time
from tqdm import tqdm

from util import helpers
from models.wide_resnet_duq import Wide_ResNet_DUQ
import config as cf

best_acc = 0


def bce_loss_fn(y_pred, y):
    bce = F.binary_cross_entropy(y_pred, y, reduction="sum").div(
        10 * y_pred.shape[0]
    )
    return bce


def output_transform_bce(output):
    y_pred, y, x = output

    y = F.one_hot(y, 10).float()

    return y_pred, y


def output_transform_acc(output):
    y_pred, y, x = output

    return y_pred, y


def output_transform_gp(output):
    y_pred, y, x = output

    return x, y_pred


def calc_gradients_input(x, y_pred):
    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]

    gradients = gradients.flatten(start_dim=1)

    return gradients


def calc_gradient_penalty(x, y_pred):
    gradients = calc_gradients_input(x, y_pred)

    # L2 norm
    grad_norm = gradients.norm(2, dim=1)

    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()

    return gradient_penalty


def eval(path_in, path_out, model, testloader, oodloader, use_cuda=True, samples=10, verbose=True):
    pass


""" Default values from the DUQ implementation at:
https://github.com/y0ast/deterministic-uncertainty-quantification
"""


def train(trainloader, validationloader, num_classes=10, epochs=5, batch_size=128, centroid_size=512, model_output_size=512, lr=0.05, gradient_penalty=0.5, gamma=0.999, length_scale=0.1, weight_decay=5e-4, use_cuda=True):
    save_point = './checkpoint/cifar10'+os.sep
    input_size = 32  # cifar 10
    model = Wide_ResNet_DUQ(28, 10, 0.3, num_classes, input_size,
                            centroid_size, model_output_size, length_scale, gamma)
    file_name = 'wide-resnet-28x10-DUQ-{}_{}_{}_{}'.format(
        length_scale, gradient_penalty, gamma, centroid_size)
    print('| Building model type [' + file_name + ']...')

    if use_cuda:
        model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 50, 75], gamma=0.2
    )

    def train_epoch(model, epoch, scheduler):
        model.train()
        model.training = True
        train_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()

        print('\n=> Current LR: {}'.format(scheduler.get_last_lr()))
        print('\n=> Training Epoch #%d' % (epoch))
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            x, y = inputs.cuda(), targets.cuda()

            if gradient_penalty > 0:
                x.requires_grad_(True)

            z, y_pred = model(x)
            y = F.one_hot(y, 10).float()

            loss = bce_loss_fn(y_pred, y)

            if gradient_penalty > 0:
                loss += gradient_penalty * calc_gradient_penalty(x, y_pred)

            loss.backward()
            optimizer.step()

            x.requires_grad_(False)

            with torch.no_grad():
                model.eval()
                model.update_embeddings(x, y)

            # print('DEBUG: LR: {}, Batch Shape: {}, {}\ny: {}\nloss: {}'.format(
            #    scheduler.get_last_lr(), inputs.shape, targets.shape, y.shape, loss.shape))

            train_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            targets = targets.cuda()
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            #print('Correct: {}, total: {}'.format(correct, total))

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                             % (epoch, epochs, batch_idx+1,
                                (len(trainloader.dataset)//batch_size)+1, loss.item(), 100.*correct/total))
            sys.stdout.flush()

    def validate_epoch(model, epoch):
        global best_acc
        model.eval()
        model.training = False
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(validationloader):
            model.eval()

            x, y = inputs.cuda(), targets.cuda()

            x.requires_grad_(True)

            z, y_pred = model(x)
            y = F.one_hot(y, 10).float()

            loss = bce_loss_fn(y_pred, y)

            test_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            targets = targets.cuda()
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model is reached
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %
              (epoch, loss.item(), acc))

        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
            # save the state with the number of the run
            torch.save(model.state_dict(), save_point+file_name+'.pth')
            best_acc = acc

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(epochs))
    print('| Initial Learning Rate = {}'.format(lr))
    print('| Optimizer = SGD')

    elapsed_time = 0
    for epoch in tqdm(range(0, epochs)):
        start_time = time.time()

        train_epoch(model, epoch, scheduler)
        validate_epoch(model, epoch)
        scheduler.step()

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

    print('* Validation results : Acc@1 = %.2f%%' % (best_acc))
    with open((save_point+file_name+'.txt'), 'w') as f:
        f.write('Run: {}\nValidation Accuracy: {}\nDataset: {}'.format(
            experiment_run, best_acc, args.dataset))
    print('| Saved all results to file. Training done.')
