# -*- coding: utf-8 -*-
"""
@Project: MachineLearning
@File   : muilti_classification.py
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



def load_data(path, viz=False):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector
    X = data.get('X')  # (5000,400)
    X = np.array([im.reshape((20, 20)).T for im in X])  # 转置，图片方向改正
    X = np.array([im.reshape(400) for im in X])
    if viz:
        plot_an_image(X)
        plot_100_image(X)
        plt.show()
    return X, y


def plot_an_image(X):
    pick_one = np.random.randint(0, 5000)
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))


def plot_100_image(X):
    size = int(np.sqrt(X.shape[1]))
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 100*400
    sample_images = X[sample_idx, :]
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


def data_precess(X, y):
    # 样本X增加x0
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # 插入了第一列（全部为1）
    # 真值y向量化
    y_matrix = []
    for k in range(1, 11):
        y_matrix.append((y == k).astype(int))  # 见配图 "向量化标签.png"
    # last one is k==10, it's digit 0, bring it to the first position，最后一列k=10，都是0，把最后一列放到第一列
    y_matrix = [y_matrix[-1]] + y_matrix[:-1]
    y = np.array(y_matrix)
    return X, y


def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))


def regularized_cost(theta, X, y, l=1):
    '''you don't penalize theta_0'''
    theta_j1_to_n = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

    return cost(theta, X, y) + regularized_term


def gradient(theta, X, y):
    '''just 1 batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)


def regularized_gradient(theta, X, y, l=1):
    '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(X, y, l=1):
    """generalized logistic regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    theta = np.zeros(X.shape[1])

    # train it
    res = opt.minimize(fun=regularized_cost,  # 需要最小化的cost function
                       x0=theta,    # 初始迭代点
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,  # cost function对应的雅克比矩阵
                       options={'disp': True})
    # get trained parameters
    final_theta = res.x

    return final_theta


def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)


if __name__ == '__main__':
    X_raw, y_raw = load_data(DATA_PATH, False)
    X, y = data_precess(X_raw, y_raw)
    # print(X.shape)
    # print(y[0].shape, y[0])
    # 单分类
    # theta = logistic_regression(X, y[0])
    # print(theta)
    # y_pred = predict(X, theta)
    # print(np.mean(y[0] == y_pred))
    # 多分类
    k_theta = np.array([logistic_regression(X, y[k]) for k in range(10)])
    print(k_theta.shape)
    # 预测
    prob_matrix = sigmoid(X @ k_theta.T)
    np.set_printoptions(suppress=True)
    y_pred = np.argmax(prob_matrix, axis=1)  # 返回沿轴axis最大值的索引，axis=1代表行
    print(y_pred)
    y_raw[y_raw==10] = 0
    print(y_raw)
    print(classification_report(y_raw, y_pred))