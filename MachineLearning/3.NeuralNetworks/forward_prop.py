# -*- coding: utf-8 -*-
"""
@Project: MachineLearning
@File   : forward_prop.py
@Author : Zhang P.H
@Date   : 2021/9/26
@Desc   :
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report  # 这个包是评价报告

DATA_PATH = '../Data/ex3data1.mat'
THETA_PATH = '../Data/ex3weights.mat'


def load_data(path):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector
    X = data.get('X')  # (5000,400)
    # X = np.array([im.reshape((20, 20)).T for im in X])  # 转置，图片方向改正
    # X = np.array([im.reshape(400) for im in X])
    return X, y


def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']

def sigmoid(z):
    """
    sigmoid函数映射
    :param z: 实数 或 ndarray
    :return: 实数 或 ndarray
    """
    return 1 / (1 + np.exp(-z))

if __name__ == '__main__':
    X, y = load_data(DATA_PATH)
    theta1, theta2 = load_weight(THETA_PATH)
    a1 = np.insert(X, 0, 1, axis=1)
    z2 = a1 @ theta1.T
    z2 = np.insert(z2, 0, 1, axis=1)
    a2 = sigmoid(z2)
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)
    y_pred = np.argmax(a3, axis=1)+1  # 5000个样本的预测结果
    print(classification_report(y, y_pred))
