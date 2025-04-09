import copy
import torch
device = "cuda"


def WFedAvg(w):

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
    return w_avg
