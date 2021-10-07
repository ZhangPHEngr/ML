# -*- coding: utf-8 -*-
"""
@Project: MachineLearning
@File   : back_prop.py
@Author : Zhang P.H
@Date   : 2021/9/27
@Desc   :
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report  # 这个包是评价报告

DATA_PATH = '../Data/ex4data1.mat'
THETA_PATH = '../Data/ex3weights.mat'


def load_data(path, transpose=False, viz=False):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector
    X = data.get('X')  # (5000,400)
    if transpose:
        X = np.array([im.reshape((20, 20)).T for im in X])  # 转置，图片方向改正
        X = np.array([im.reshape(400) for im in X])
    if viz:
        plot_100_image(X)
        plt.show()
    return X, y


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


def expand_y(y):
    #     """expand 5000*1 into 5000*10
    #     where y=10 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]: ndarray
    #     """
    res = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i - 1] = 1

        res.append(y_array)

    return np.array(res)


def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


# 序列化和反序列化，便于传参
def serialize(a, b):
    return np.concatenate((np.ravel(a), np.ravel(b)))


def deserialize(seq):
    #     """into ndarray of (25, 401), (10, 26)"""
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y):
    m = X.shape[0]  # get the svm size m
    _, _, _, _, h = feed_forward(theta, X)

    # np.multiply is pairwise operation
    pair_computation = -np.multiply(y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))

    return pair_computation.sum() / m


def regularized_cost(theta, X, y, l=1):
    """the first column of t1 and t2 is intercept theta, ignore them when you do regularization"""
    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]

    reg_t1 = (l / (2 * m)) * np.power(t1[:, 1:], 2).sum()  # this is how you ignore first col
    reg_t2 = (l / (2 * m)) * np.power(t2[:, 1:], 2).sum()

    return cost_function(theta, X, y) + reg_t1 + reg_t2


def feed_forward(theta, X):
    """apply to architecture 400+1 * 25+1 *10
    X: 5000 * 401
    """

    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]
    a1 = X  # 5000 * 401

    z2 = a1 @ t1.T  # 5000 * 25
    a2 = np.insert(sigmoid(z2), 0, np.ones(m), axis=1)  # 5000*26

    z3 = a2 @ t2.T  # 5000 * 10
    h = sigmoid(z3)  # 5000*10, this is h_theta(X)

    return a1, z2, a2, z3, h  # you need all those for backprop


def gradient(theta, X, y):
    # initialize
    t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
    m = X.shape[0]

    delta1 = np.zeros(t1.shape)  # (25, 401)
    delta2 = np.zeros(t2.shape)  # (10, 26)

    a1, z2, a2, z3, h = feed_forward(theta, X)

    for i in range(m):
        a1i = a1[i, :]  # (1, 401)
        a2i = a2[i, :]  # (1, 26)

        hi = h[i, :]  # (1, 10)
        yi = y[i, :]  # (1, 10)

        d3i = hi - yi  # (1, 10)
        d2i = np.multiply(t2.T @ d3i, a2i*(1-a2i))  # (1, 26)

        # careful with np vector transpose
        delta2 += np.matrix(d3i).T @ np.matrix(a2i)  # (1, 10).T @ (1, 26) -> (10, 26)
        delta1 += np.matrix(d2i[1:]).T @ np.matrix(a1i)  # (1, 25).T @ (1, 401) -> (25, 401)

    delta1 = delta1 / m
    delta2 = delta2 / m

    return serialize(delta1, delta2)

def regularized_gradient(theta, X, y, l=1):
    """don't regularize theta of bias terms"""
    m = X.shape[0]
    delta1, delta2 = deserialize(gradient(theta, X, y))
    t1, t2 = deserialize(theta)

    t1[:, 0] = 0
    reg_term_d1 = (l / m) * t1
    delta1 = delta1 + reg_term_d1

    t2[:, 0] = 0
    reg_term_d2 = (l / m) * t2
    delta2 = delta2 + reg_term_d2

    return serialize(delta1, delta2)


def random_init(size):
    return np.random.uniform(-0.12, 0.12, size)

def nn_training(X, y):
    """regularized version
    the architecture is hard coded here... won't generalize
    """
    init_theta = random_init(10285)  # 25*401 + 10*26

    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400})
    return res

def show_accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(theta, X)

    y_pred = np.argmax(h, axis=1) + 1

    print(classification_report(y, y_pred))

def plot_hidden_layer(theta):
    """
    theta: (10285, )
    """
    final_theta1, _ = deserialize(theta)
    hidden_layer = final_theta1[:, 1:]  # ger rid of bias term theta

    fig, ax_array = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, figsize=(5, 5))

    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(hidden_layer[5 * r + c].reshape((20, 20)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

if __name__ == '__main__':
    # 加载并预处理数据
    X_raw, y_raw = load_data(DATA_PATH, transpose=False)
    X = np.insert(X_raw, 0, 1, axis=1)  # 增加全部为1的一列
    y = expand_y(y_raw)  # y展成向量,例如y=9 则向量化后为[0 0 0 0 0 0 0 0 1 0]
    t1, t2 = load_weight(THETA_PATH)
    theta = serialize(t1, t2)  # 扁平化参数，25*401+10*26=10285
    # 前向传播 后向传播算法里已经包含
    print(cost_function(theta, X, y))
    print(regularized_cost(theta, X, y))
    # 后向传播
    d1, d2 = deserialize(gradient(theta, X, y))
    print(d1.shape, d2.shape)
    res = nn_training(X, y)  # 慢
    print(res)
    final_theta = res.x
    show_accuracy(final_theta, X, y_raw)
    plot_hidden_layer(final_theta)
    plt.show()
