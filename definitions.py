import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
import random

def sigmoid(value):
    return 1 / (1 + np.exp(-value))

def leakyrelu(value):
    return np.max((value * 0.01, value), axis=0)

def derivativeleakyrelu(value):
    return np.where(value > 0, 1, 0.01)

def isworking(model, data, expected):
    model.calculatelayers(data)
    if np.argmax(model.activations[-1]) == expected:
        return True
    else:
        return False