import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix


def generate_3d_agreement(gt, pred_1, pred_2, k):
    # create tensor of agreement by ground truth
    t_mat = np.empty((k, k, k), dtype=int)
    for k0 in range(k):
        for k1 in range(k):
            for k2 in range(k):
                t_mat[k0, k1, k2] = np.sum((gt == k0) & (pred_1 == k1) & (pred_2 == k2))
    # agreement matrix
    a_mat = np.sum(t_mat, axis=0)
    c1 = np.sum(t_mat, axis=2)
    c2 = np.sum(t_mat, axis=1)
    return t_mat, a_mat, c1, c2

def generate_error_agreement(gt, pred_1, pred_2, k):
    # create tensor of agreement by ground truth
    ea_mat = np.zeros((k, k), dtype=int)
    for n, (pred_i, pred_j) in enumerate(zip(pred_1, pred_2)):
        if ((pred_i != gt[n]) and (pred_j != gt[n])):
            ea_mat[pred_i, pred_j] += 1
    return ea_mat

def get_confusion_mat(gt, pred, labels=None):
    return confusion_matrix(gt, pred, labels=labels)

def get_confusion_errors(gt, pred, labels=None):
    C =  confusion_matrix(gt, pred, labels=labels)
    return C - np.diag(C.diagonal())





