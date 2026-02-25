import numpy as np

def softmax(value):
    exp = np.exp(value)
    return exp / np.sum(exp, axis = 0, keepdims=True)

def leakyrelu(value, param):
    return np.where(value > 0, value, param * value)

def derivativeleakyrelu(value, param):
    return np.where(value > 0, 1, param)