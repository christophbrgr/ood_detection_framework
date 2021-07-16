import torch
import torch.nn as nn
import numpy as np 
from torch.autograd import Variable
import sys
import os

from util import helpers

def eval(path_in, path_out, nets, testloader, oodloader, use_cuda=True, samples=10, verbose=True, save_dir=None):
    f1 = open(path_in, 'w')
    f2 = open(path_out, 'w')
    # to enable Monte Carlo Dropout
    for net in nets: 
        net.eval()
        net.training = False
    print('| Running Deepensemble with {} members'.format(len(nets)))
    correct = 0
    total = 0
    confidence_list = []
    correct_list = []
    predicted_list = []
    labels_list = []
    print('| Classification confidence for ID is saved at: {}'.format(path_in))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            #outputs = net(inputs)
            # this is the OOD magic
            #nnOutputs = helpers.softmax(outputs)
            labels_list.append(targets.data)
            out = [helpers.softmax(net(inputs)) for net in nets]
            out_stack = np.stack(out, axis=2) # shape should be 128,10,10
            #print(out_stack.shape)
            nnOutputs = np.mean(out_stack, axis=2) # shape should now be 128,10
            #print(nnOutputs.shape)
            for k in range(len(inputs)):
                f1.write("{}\n".format(np.max(nnOutputs[k])))
                confidence_list.append(np.max(nnOutputs[k]))
            predicted = np.argmax(nnOutputs, 1)
            total += targets.size(0)
            correct += np.sum(np.equal(predicted, targets.data.cpu().numpy()))
            correct_list.extend(np.equal(predicted, targets.data.cpu().numpy()).tolist())
            predicted_list.extend(predicted.tolist())
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
        acc_list = (sum(correct_list)/len(correct_list))
        if save_dir:
            with open(os.path.join(save_dir, 'deepensemble_targets.txt'), 'w') as f:
                for item in labels_list:
                    f.write('{}\n'.format(item.cpu().numpy()[0]))
            with open(os.path.join(save_dir, 'deepensemble_pred.txt'), 'w') as f:
                for item in predicted_list:
                    f.write('{}\n'.format(item))
            with open(os.path.join(save_dir, 'deepensemble_correct.txt'), 'w') as f:
                for item in correct_list:
                    f.write('{}\n'.format(item))
            with open(os.path.join(save_dir, 'deepensemble_confidence.txt'), 'w') as f:
                for item in confidence_list:
                    f.write('{}\n'.format(item))
    print('| Classification confidence for OOD is saved at: {}'.format(path_out))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(oodloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            out = [helpers.softmax(net(inputs)) for net in nets]
            out_stack = np.stack(out, axis=2) # shape should be 128,10,10
            #print(out_stack.shape)
            nnOutputs = np.mean(out_stack, axis=2) # shape should now be 128,10
            #print(nnOutputs.shape)
            for k in range(len(inputs)):
                f2.write("{}\n".format(np.max(nnOutputs[k])))

def eval_cifar10(path_in, path_out, nets, testloader, oodloader, use_cuda=True, samples=10, verbose=True, save_dir=None):
    f1 = open(path_in, 'w')
    f2 = open(path_out, 'w')
    # to enable Monte Carlo Dropout
    for net in nets: 
        net.eval()
        net.training = False
    print('| Running Deepensemble with {} members'.format(len(nets)))
    correct = 0
    total = 0
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
            out = [helpers.softmax(net(inputs)) for net in nets]
            out_stack = np.stack(out, axis=2) # shape should be 128,10,10
            #print(out_stack.shape)
            nnOutputs = np.mean(out_stack, axis=2) # shape should now be 128,10
            #print(nnOutputs.shape)
            for k in range(len(inputs)):
                f1.write("{}\n".format(np.max(nnOutputs[k])))
                confidence_list.append(np.max(nnOutputs[k]))
            predicted = np.argmax(nnOutputs, 1)
            total += targets.size(0)
            correct += np.sum(np.equal(predicted, targets.data.cpu().numpy()))
            correct_list.extend(np.equal(predicted, targets.data.cpu().numpy()).tolist())
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
        acc_list = (sum(correct_list)/len(correct_list))
        if save_dir:
            with open(os.path.join(save_dir, 'de_correct.txt'), 'w') as f:
                for item in correct_list:
                    f.write('{}\n'.format(item))
            with open(os.path.join(save_dir, 'de_confidence.txt'), 'w') as f:
                for item in confidence_list:
                    f.write('{}\n'.format(item))

    print('| Classification confidence for OOD is saved at: {}'.format(path_out))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(oodloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            out = [helpers.softmax(net(inputs)) for net in nets]
            out_stack = np.stack(out, axis=2) # shape should be 128,10,10
            #print(out_stack.shape)
            nnOutputs = np.mean(out_stack, axis=2) # shape should now be 128,10
            #print(nnOutputs.shape)
            for k in range(len(inputs)):
                f2.write("{}\n".format(np.max(nnOutputs[k])))

def train():
    pass

