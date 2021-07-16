import numpy as np
import sklearn.metrics as skm

import torch
from torch import nn, optim
from torch.nn import functional as F


def get_metrics(path_id_confidence, path_ood_confidence, verbose=True, normalized=True):
  """ Returns most common metrics (AUC, FPR, TPR) for comparing OOD vs ID inputs.
  Assumes that values are probabilities/confidences between 0 and 1 as default. 
  If not, please set normalized to False
  """
  id = np.loadtxt(path_id_confidence, delimiter='\n')
  ood = np.loadtxt(path_ood_confidence, delimiter='\n')
  if verbose:
    print('Mean confidence OOD: {}, Median: {}, Length: {}'.format(np.mean(ood), np.median(ood), len(ood)))
    print('Mean confidence ID: {}, Median: {}, Length: {}'.format(np.mean(id), np.median(id), len(id)))
  id_l = np.ones(len(id))
  ood_l = np.zeros(len(ood))
  true_labels = np.concatenate((id_l, ood_l))
  pred_probs = np.concatenate((id, ood))
  assert(len(true_labels) == len(pred_probs))
  if not normalized:
    # use unity based normalization to also catch negative values
    pred_probs = (pred_probs - np.min(pred_probs))/(np.max(pred_probs) - np.min(pred_probs))
  fpr, tpr, thresholds = skm.roc_curve(y_true = true_labels, y_score = pred_probs, pos_label = 1) #positive class is 1; negative class is 0
  auroc = skm.auc(fpr, tpr)
  precision, recall, _ = skm.precision_recall_curve(true_labels, pred_probs)
  aucpr = skm.auc(recall, precision)
  if verbose:
    print('AUROC: {}'.format(auroc))
  return auroc, aucpr, fpr, tpr



class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=20):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


# second implementation as sanity check
# py is the model output, y_test are the labels
def ece_score(py, y_test, n_bins=20):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)
