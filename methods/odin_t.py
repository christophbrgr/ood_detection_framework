import torch
import torch.nn as nn
import numpy as np 
from torch.autograd import Variable


from util import helpers
from util.metrics import ECELoss, ece_score
import sklearn.metrics as skm
import os

def eval(path_in, path_out, net, testloader, oodloader, use_cuda=True, noiseMagnitude = 0.0014, temper = 10, save_dir=None):
    f1 = open(path_in, 'w')
    f2 = open(path_out, 'w')
    criterion = nn.CrossEntropyLoss()
    ece_criterion = ECELoss().cuda()
    net.eval()
    net.training = False
    correct = 0
    total = 0
    logits_list = []
    labels_list = []
    confidence_list = []
    correct_list = []
    predicted_list = []
    print('| Classification confidence for ID ODIN is saved at: {}'.format(path_in))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = Variable(inputs.cuda(), requires_grad=True), Variable(targets.cuda())
        outputs = net(inputs)
        # get our confidence
        nnOutputs = helpers.softmax(outputs, temper=temper)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        with torch.no_grad():
            logits_list.append(outputs.data)
            # nnOutputs = outputs.data.cuda()
            # nnOutputs = nnOutputs - torch.max(nnOutputs, axis=1, keepdims=True)[0]
            #logits_list.append(torch.div(nnOutputs, 1000)) # ODIN temper
            labels_list.append(targets.data)

        for k in range(len(inputs)):
            f1.write("{}\n".format(np.max(nnOutputs[k])))
            confidence_list.append(np.max(nnOutputs[k]))
        
        correct_list.extend(predicted.eq(targets.data).cpu().tolist())
        predicted_list.extend(predicted.cpu().tolist())

    with torch.no_grad():
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
        ece = ece_criterion(logits, labels)
    acc = 100.*correct/total
    acc_list = (sum(correct_list)/len(correct_list))
    if save_dir:
        with open(os.path.join(save_dir, 'odin_t_targets.txt'), 'w') as f:
            for item in labels_list:
                f.write('{}\n'.format(item.cpu().numpy()[0]))
        with open(os.path.join(save_dir, 'odin_t_pred.txt'), 'w') as f:
            for item in predicted_list:
                f.write('{}\n'.format(item))
        with open(os.path.join(save_dir, 'odin_t_correct.txt'), 'w') as f:
            for item in correct_list:
                f.write('{}\n'.format(item))
        with open(os.path.join(save_dir, 'odin_t_confidence.txt'), 'w') as f:
            for item in confidence_list:
                f.write('{}\n'.format(item))
    # calculate AUROC for classifcation accuracy
    fpr, tpr, _ = skm.roc_curve(y_true = correct_list, y_score = confidence_list, pos_label = 1) #positive class is 1; negative class is 0
    auroc_classification = skm.auc(fpr, tpr)
    print("| Test Result\tAcc@1: %.2f%%" %(acc))
    print(f'| ECE: {ece.item()}')
    print(f'| ECE v2: {ece_score(logits.cpu(), labels.cpu())}')
    print(f'| Acc list: {acc_list}')
    print(f'| AUROC classification: {auroc_classification}')

    print('| Classification confidence for OOD ODIN is saved at: {}'.format(path_out))
    for batch_idx, (inputs, targets) in enumerate(oodloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        outputs = net(inputs)
        # get our confidence
        nnOutputs = helpers.softmax(outputs, temper=temper)
        # Using temperature scaling
        for k in range(len(inputs)):
            f2.write("{}\n".format(np.max(nnOutputs[k])))

def eval_cifar10(path_in, path_out, net, testloader, oodloader, use_cuda=True, noiseMagnitude = 0.0014, temper = 1000, save_dir=None):
    f1 = open(path_in, 'w')
    f2 = open(path_out, 'w')
    criterion = nn.CrossEntropyLoss()
    ece_criterion = ECELoss().cuda()
    net.eval()
    net.training = False
    correct = 0
    total = 0
    logits_list = []
    labels_list = []
    confidence_list = []
    correct_list = []
    predicted_list = []
    print('| Classification confidence for ID ODIN is saved at: {}'.format(path_in))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = Variable(inputs.cuda(), requires_grad=True), Variable(targets.cuda())
        outputs = net(inputs)
        # get our confidence
        nnOutputs = helpers.softmax(outputs, temper=temper)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        with torch.no_grad():
            logits_list.append(outputs.data)
            # nnOutputs = outputs.data.cuda()
            # nnOutputs = nnOutputs - torch.max(nnOutputs, axis=1, keepdims=True)[0]
            #logits_list.append(torch.div(nnOutputs, 1000)) # ODIN temper
            labels_list.append(targets.data)

        for k in range(len(inputs)):
            f1.write("{}\n".format(np.max(nnOutputs[k])))
            confidence_list.append(np.max(nnOutputs[k]))
        
        correct_list.extend(predicted.eq(targets.data).cpu().tolist())
        predicted_list.extend(predicted.cpu().tolist())

    with torch.no_grad():
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
        ece = ece_criterion(logits, labels)
        labels_list = torch.cat(labels_list).cpu().tolist()
    acc = 100.*correct/total
    acc_list = (sum(correct_list)/len(correct_list))
    if save_dir:
        with open(os.path.join(save_dir, 'odin_t_targets_cifar10.txt'), 'w') as f:
            for item in labels_list:
                f.write('{}\n'.format(item))
        with open(os.path.join(save_dir, 'odin_t_pred_cifar10.txt'), 'w') as f:
            for item in predicted_list:
                f.write('{}\n'.format(item))
        with open(os.path.join(save_dir, 'odin_t_correct_cifar10.txt'), 'w') as f:
            for item in correct_list:
                f.write('{}\n'.format(item))
        with open(os.path.join(save_dir, 'odin_t_confidence_cifar10.txt'), 'w') as f:
            for item in confidence_list:
                f.write('{}\n'.format(item))
    # calculate AUROC for classifcation accuracy
    fpr, tpr, _ = skm.roc_curve(y_true = correct_list, y_score = confidence_list, pos_label = 1) #positive class is 1; negative class is 0
    auroc_classification = skm.auc(fpr, tpr)
    print("| Test Result\tAcc@1: %.2f%%" %(acc))
    print(f'| ECE: {ece.item()}')
    print(f'| ECE v2: {ece_score(logits.cpu(), labels.cpu())}')
    print(f'| Acc list: {acc_list}')
    print(f'| AUROC classification: {auroc_classification}')

    print('| Classification confidence for OOD ODIN is saved at: {}'.format(path_out))
    for batch_idx, (inputs, targets) in enumerate(oodloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        outputs = net(inputs)
        # get our confidence
        nnOutputs = helpers.softmax(outputs, temper=temper)
        # Using temperature scaling
        for k in range(len(inputs)):
            f2.write("{}\n".format(np.max(nnOutputs[k])))


def train():
    pass

