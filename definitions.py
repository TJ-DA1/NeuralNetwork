import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
import random

def softmax(value):
    exp = np.exp(value)
    return exp / np.sum(exp, axis = 0, keepdims=True)

def leakyrelu(value, param):
    return np.where(value > 0, value, param * value)

def derivativeleakyrelu(value, param):
    return np.where(value > 0, 1, param)

def isworking(model, data, expected):
    model.calculatelayers(data)
    if np.argmax(model.activations[-1]) == expected:
        return True
    else:
        return False