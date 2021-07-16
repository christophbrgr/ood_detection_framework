import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import sys
from tqdm import tqdm

from util import helpers
from util.mahalanobis_lib import get_Mahalanobis_score, sample_estimator, sample_estimator_cifar10
from util.metrics import get_metrics


def eval(path_in, path_out, model, trainloader, testloader, oodloader, use_cuda=True, verbose=True, magnitude=0.0, num_classes=10, precision=None, sample_mean=None, feature_list=None, save_dir=None):
    # loading data sets

    model.eval()
    model.cuda()
    temp_x = torch.rand(2, 3, 224, 224)
    temp_x = Variable(temp_x.cuda())
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)

    if precision == None or sample_mean == None or feature_list == None:
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
        sample_mean, precision = sample_estimator(
            model, num_classes=num_classes, feature_list=feature_list, train_loader=trainloader)

    f1 = open(path_in, 'w')
    f2 = open(path_out, 'w')

########################################In-distribution###########################################
    print("Processing in-distribution images")
    count = 0
    for j, data in tqdm(enumerate(testloader)):

        images, _, _ = data
        batch_size = images.shape[0]

        inputs = images.cuda()

        Mahalanobis_scores = get_Mahalanobis_score(
            model, inputs, num_classes, sample_mean, precision, num_output, magnitude)

        for k in range(batch_size):
            f1.write("{}\n".format(Mahalanobis_scores[k, 0]))

        count += batch_size
        # print("{:4}/{:4} images processed.".format(count, len(testloader.dataset)))


###################################Out-of-Distributions#####################################
    print("Processing out-of-distribution images")

    count = 0

    for j, data in tqdm(enumerate(oodloader)):

        images, labels, _ = data
        batch_size = images.shape[0]

        inputs = images.cuda()

        Mahalanobis_scores = get_Mahalanobis_score(
            model, inputs, num_classes, sample_mean, precision, num_output, magnitude)

        for k in range(batch_size):
            f2.write("{}\n".format(Mahalanobis_scores[k, 0]))

        count += batch_size
        # print("{:4}/{:4} images processed.".format(count, len(oodloader.dataset)))

    f1.close()
    f2.close()
    auroc, aucpr, _, _ = get_metrics(path_in, path_out)

    print('Mahalanobis AUROC: {}, AUCPR: {}'.format(auroc, aucpr))
    return

def eval_cifar10(path_in, path_out, model, trainloader, testloader, oodloader, use_cuda=True, verbose=True, magnitude=0.0, num_classes=10, precision=None, sample_mean=None, feature_list=None, save_dir=None):
    # loading data sets

    model.eval()
    model.cuda()
    temp_x = torch.rand(2, 3, 32, 32)
    temp_x = Variable(temp_x.cuda())
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)

    if precision == None or sample_mean == None or feature_list == None:
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
        sample_mean, precision = sample_estimator_cifar10(
            model, num_classes=num_classes, feature_list=feature_list, train_loader=trainloader)

    f1 = open(path_in, 'w')
    f2 = open(path_out, 'w')

########################################In-distribution###########################################
    print("Processing in-distribution images")
    count = 0
    for j, data in tqdm(enumerate(testloader)):

        images, _ = data
        batch_size = images.shape[0]

        inputs = images.cuda()

        Mahalanobis_scores = get_Mahalanobis_score(
            model, inputs, num_classes, sample_mean, precision, num_output, magnitude)

        for k in range(batch_size):
            f1.write("{}\n".format(Mahalanobis_scores[k, 0]))

        count += batch_size
        # print("{:4}/{:4} images processed.".format(count, len(testloader.dataset)))


###################################Out-of-Distributions#####################################
    print("Processing out-of-distribution images")

    count = 0

    for j, data in tqdm(enumerate(oodloader)):

        images, labels = data
        batch_size = images.shape[0]

        inputs = images.cuda()

        Mahalanobis_scores = get_Mahalanobis_score(
            model, inputs, num_classes, sample_mean, precision, num_output, magnitude)

        for k in range(batch_size):
            f2.write("{}\n".format(Mahalanobis_scores[k, 0]))

        count += batch_size
        # print("{:4}/{:4} images processed.".format(count, len(oodloader.dataset)))

    f1.close()
    f2.close()
    auroc, aucpr, _, _ = get_metrics(path_in, path_out)

    print('Mahalanobis AUROC: {}, AUCPR: {}'.format(auroc, aucpr))
    return

def train():
    pass
