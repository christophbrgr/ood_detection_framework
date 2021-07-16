import torch
import torch.nn as nn
import numpy as np 
from torch.autograd import Variable
import sys

from util import helpers
from util.metrics import ECELoss, ece_score

def enable_dropout(model, dropout_rate=0.3):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        # print
        if m.__class__.__name__.startswith('Dropout'):
            # m.p=dropout_rate # this does not work :-/ 
            m.train()

def eval(path_in, path_out, net, testloader, oodloader, use_cuda=True, samples=10, verbose=True, save_dir=None):
    f1 = open(path_in, 'w')
    f2 = open(path_out, 'w')
    ece_criterion = ECELoss().cuda()
    # to enable Monte Carlo Dropout
    net.eval()
    enable_dropout(net)
    correct = 0
    total = 0
    logits_list = []
    labels_list = []
    confidence_list = []
    correct_list = []
    print('| Classification confidence for ID is saved at: {}'.format(path_in))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            #outputs = net(inputs)
            # this is the OOD magic
            #nnOutputs = helpers.softmax(outputs)
            out = [helpers.softmax(net(inputs)) for _ in range(samples)]
            out_stack = np.stack(out, axis=2) # shape should be 128,10,10
            #print(out_stack.shape)
            nnOutputs = np.mean(out_stack, axis=2) # shape should now be 128,10
            #print(nnOutputs.shape)
            for k in range(len(inputs)):
                f1.write("{}\n".format(np.max(nnOutputs[k])))
            predicted = np.argmax(nnOutputs, 1)
            total += targets.size(0)
            correct += np.sum(np.equal(predicted, targets.data.cpu().numpy()))
            #correct += predicted.eq(targets.data).cpu().sum()
            # Calculating mean across multiple MCD forward passes 
            mean = nnOutputs # shape (n_samples, n_classes)

            # Calculating variance across multiple MCD forward passes 
            variance = np.var(out_stack, axis=0) # shape (n_samples, n_classes)

            epsilon = sys.float_info.min
            # Calculating entropy across multiple MCD forward passes 
            entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)

            # Calculating mutual information across multiple MCD forward passes 
            #mutual_info = entropy - np.mean(np.sum(-out_stack*np.log(out_stack + epsilon),
            #                                        axis=-1), axis=0) # shape (n_samples,)

        acc = 100.*correct/total
        print("| Test Result\tAcc@1: %.2f%%" %(acc))
    print('| Classification confidence for OOD is saved at: {}'.format(path_out))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(oodloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            out = [helpers.softmax(net(inputs)) for _ in range(samples)]
            out_stack = np.stack(out, axis=2) # shape should be 128,10,10
            #print(out_stack.shape)
            nnOutputs = np.mean(out_stack, axis=2) # shape should now be 128,10
            #print(nnOutputs.shape)
            for k in range(len(inputs)):
                f2.write("{}\n".format(np.max(nnOutputs[k])))

def eval_cifar10(path_in, path_out, net, testloader, oodloader, use_cuda=True, samples=10, verbose=True, save_dir=None):
    f1 = open(path_in, 'w')
    f2 = open(path_out, 'w')
    ece_criterion = ECELoss().cuda()
    # to enable Monte Carlo Dropout
    net.eval()
    enable_dropout(net)
    correct = 0
    total = 0
    logits_list = []
    labels_list = []
    confidence_list = []
    correct_list = []
    print('| Classification confidence for ID is saved at: {}'.format(path_in))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            #outputs = net(inputs)
            # this is the OOD magic
            #nnOutputs = helpers.softmax(outputs)
            out = [helpers.softmax(net(inputs)) for _ in range(samples)]
            out_stack = np.stack(out, axis=2) # shape should be 128,10,10
            #print(out_stack.shape)
            nnOutputs = np.mean(out_stack, axis=2) # shape should now be 128,10
            #print(nnOutputs.shape)
            for k in range(len(inputs)):
                f1.write("{}\n".format(np.max(nnOutputs[k])))
            predicted = np.argmax(nnOutputs, 1)
            total += targets.size(0)
            correct += np.sum(np.equal(predicted, targets.data.cpu().numpy()))
            #correct += predicted.eq(targets.data).cpu().sum()
            # Calculating mean across multiple MCD forward passes 
            mean = nnOutputs # shape (n_samples, n_classes)

            # Calculating variance across multiple MCD forward passes 
            variance = np.var(out_stack, axis=0) # shape (n_samples, n_classes)

            epsilon = sys.float_info.min
            # Calculating entropy across multiple MCD forward passes 
            entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)

            # Calculating mutual information across multiple MCD forward passes 
            #mutual_info = entropy - np.mean(np.sum(-out_stack*np.log(out_stack + epsilon),
            #                                        axis=-1), axis=0) # shape (n_samples,)

        acc = 100.*correct/total
        print("| Test Result\tAcc@1: %.2f%%" %(acc))
    print('| Classification confidence for OOD is saved at: {}'.format(path_out))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(oodloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            out = [helpers.softmax(net(inputs)) for _ in range(samples)]
            out_stack = np.stack(out, axis=2) # shape should be 128,10,10
            #print(out_stack.shape)
            nnOutputs = np.mean(out_stack, axis=2) # shape should now be 128,10
            #print(nnOutputs.shape)
            for k in range(len(inputs)):
                f2.write("{}\n".format(np.max(nnOutputs[k])))

def train():
    pass

