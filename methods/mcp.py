import torch
import numpy as np 
from torch.autograd import Variable

from util import helpers
from util.metrics import ECELoss, ece_score
import sklearn.metrics as skm
import os
import pandas as pd
import pickle

def eval(path_in, path_out, net, testloader, oodloader, use_cuda=True, save_dir=None):
    f1 = open(path_in, 'w')
    f2 = open(path_out, 'w')
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
    sne_embeddings = []
    print('| Classification confidence for ID is saved at: {}'.format(path_in))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs, hidden = net(inputs)
            # this is the OOD magic
            nnOutputs = helpers.softmax(outputs)
            for k in range(len(inputs)):
                f1.write("{}\n".format(np.max(nnOutputs[k])))
                confidence_list.append(np.max(nnOutputs[k]))
                sne_embeddings.append(hidden.data.cpu()[k].numpy())
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            correct_list.extend(predicted.eq(targets.data).cpu().tolist())
            predicted_list.extend(predicted.cpu().tolist())
            logits_list.append(outputs.data)
            labels_list.append(targets.data)

        
    logits = torch.cat(logits_list).cuda()
    labels = torch.cat(labels_list).cuda()
    ece = ece_criterion(logits, labels)
    if save_dir:
        with open(os.path.join(save_dir, 'mcp_sne.pkl'), 'wb') as f:
            pickle.dump(sne_embeddings, f)
        with open(os.path.join(save_dir, 'mcp_targets.txt'), 'w') as f:
            for item in labels_list:
                f.write('{}\n'.format(item.cpu().numpy()[0]))
        with open(os.path.join(save_dir, 'mcp_pred.txt'), 'w') as f:
            for item in predicted_list:
                f.write('{}\n'.format(item))
        with open(os.path.join(save_dir, 'mcp_correct.txt'), 'w') as f:
            for item in correct_list:
                f.write('{}\n'.format(item))
        with open(os.path.join(save_dir, 'mcp_confidence.txt'), 'w') as f:
            for item in confidence_list:
                f.write('{}\n'.format(item))
    acc = 100.*correct/total
    acc_list = (sum(correct_list)/len(correct_list))

    # calculate AUROC for classifcation accuracy
    fpr, tpr, _ = skm.roc_curve(y_true = correct_list, y_score = confidence_list, pos_label = 1) #positive class is 1; negative class is 0
    auroc_classification = skm.auc(fpr, tpr)
    
    print("| Test Result\tAcc@1: %.2f%%" %(acc))
    print(f'| ECE: {ece.item()}')
    print(f'| ECE v2: {ece_score(logits.cpu(), labels.cpu())}')
    print(f'| Acc list: {acc_list}')
    print(f'| AUROC classification: {auroc_classification}')

    sne_embeddings_ood = []

    print('| Classification confidence for OOD is saved at: {}'.format(path_out))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(oodloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs, hidden = net(inputs)
            # this is the OOD magic
            nnOutputs = helpers.softmax(outputs)
            for k in range(len(inputs)):
                f2.write("{}\n".format(np.max(nnOutputs[k])))
                sne_embeddings_ood.append(hidden.data.cpu()[k].numpy())
    if save_dir:
        with open(os.path.join(save_dir, 'mcp_sne_ood.pkl'), 'wb') as f:
            pickle.dump(sne_embeddings_ood, f)

def eval_cifar10(path_in, path_out, net, testloader, oodloader, use_cuda=True, save_dir=None):
    f1 = open(path_in, 'w')
    f2 = open(path_out, 'w')
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
    sne_embeddings = []
    print('| Classification confidence for ID is saved at: {}'.format(path_in))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs, hidden = net(inputs)
            # this is the OOD magic
            nnOutputs = helpers.softmax(outputs)
            for k in range(len(inputs)):
                f1.write("{}\n".format(np.max(nnOutputs[k])))
                confidence_list.append(np.max(nnOutputs[k]))
                sne_embeddings.append(hidden.data.cpu()[k].numpy())
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            correct_list.extend(predicted.eq(targets.data).cpu().tolist())
            predicted_list.extend(predicted.cpu().tolist())
            logits_list.append(outputs.data)
            labels_list.append(targets.data)
        
    logits = torch.cat(logits_list).cuda()
    labels = torch.cat(labels_list).cuda()
    labels_list = torch.cat(labels_list).cpu().tolist()
    ece = ece_criterion(logits, labels)
    if save_dir:
        with open(os.path.join(save_dir, 'mcp_sne_cifar10.pkl'), 'wb') as f:
            pickle.dump(sne_embeddings, f)
        with open(os.path.join(save_dir, 'mcp_targets_cifar10.txt'), 'w') as f:
            for item in labels_list:
                f.write('{}\n'.format(item))
        with open(os.path.join(save_dir, 'mcp_pred_cifar10.txt'), 'w') as f:
            for item in predicted_list:
                f.write('{}\n'.format(item))
        with open(os.path.join(save_dir, 'mcp_correct_cifar10.txt'), 'w') as f:
            for item in correct_list:
                f.write('{}\n'.format(item))
        with open(os.path.join(save_dir, 'mcp_confidence_cifar10.txt'), 'w') as f:
            for item in confidence_list:
                f.write('{}\n'.format(item))
    acc = 100.*correct/total
    acc_list = (sum(correct_list)/len(correct_list))

    # calculate AUROC for classifcation accuracy
    fpr, tpr, _ = skm.roc_curve(y_true = correct_list, y_score = confidence_list, pos_label = 1) #positive class is 1; negative class is 0
    auroc_classification = skm.auc(fpr, tpr)
    
    print("| Test Result\tAcc@1: %.2f%%" %(acc))
    print(f'| ECE: {ece.item()}')
    print(f'| ECE v2: {ece_score(logits.cpu(), labels.cpu())}')
    print(f'| Acc list: {acc_list}')
    print(f'| AUROC classification: {auroc_classification}')
    sne_embeddings_ood = []
    print('| Classification confidence for OOD is saved at: {}'.format(path_out))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(oodloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs, hidden = net(inputs)
            # this is the OOD magic
            nnOutputs = helpers.softmax(outputs)
            for k in range(len(inputs)):
                f2.write("{}\n".format(np.max(nnOutputs[k])))
                sne_embeddings_ood.append(hidden.data.cpu()[k].numpy())


    if save_dir:
        with open(os.path.join(save_dir, 'mcp_sne_ood_cifar10.pkl'), 'wb') as f:
            pickle.dump(sne_embeddings_ood, f)
def train():
    pass

