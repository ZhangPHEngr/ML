#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @FileName  :linear_regression_one_variable.py
# @Time      :2021/9/5 17:23
# @Author    :ZhangP.H
# Function Description:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

DATA_PATH = '../Data/ex1data2.txt'

'''
线性回归流程：
    1. 设计多项式回归模型H：
    2. 设计误差函数J:
    3. 对误差函数各参数求导dJ:
    4. 迭代计算
'''


def load_data(data_path, do_vis=False):
    data = pd.read_csv(data_path, header=None, names=['Size', 'Bedrooms', 'Price'])
    if do_vis:
        # svm.plot(kind='scatter', x='Size', y='Bedrooms', z='Price', figsize=(12, 8))
        # plt.show()
        x = data['Size'].tolist()
        y = data['Bedrooms'].tolist()
        z = data['Price'].tolist()
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x, y, z)
        # plt.show()
    # 数据预处理
    data = pre_process(data)
    data.insert(0, 'Ones', 1)  # 增加一列1便于后序特征计算  y = a_0 * x_0 + a_1 * x_1
    return data

def pre_process(data):
    # 特征缩放
    return (data - data.mean()) / data.std()
    # svm.plot(kind='scatter', x='Size', y='Price', figsize=(12, 8))
    # plt.show()

def cost_function(y, X, theta):
    """
    计算误差
    :param X: m * (n+1)  m个样本 n+1个特征
    :param y: m * 1 真值
    :param theta: (n+1) * 1 模型参数
    :return: 样本真值与假设模型输出的误差值
    """
    inner = np.power((X.dot(theta) - y), 2)
    return np.sum(inner) / (2 * len(X))


def regression(data=pd.DataFrame(), alpha=0.01, iterations=1000):
    """
    单变量线性回归
    :param data: pd
    :param alpha: 学习率
    :param iterations: 迭代次数
    :return:
    """
    n_sample = data.shape[0]
    n_feature = data.shape[1] - 1
    X = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1]).reshape(n_sample, 1)
    theta = np.zeros((n_feature, 1))
    J = list()
    for cnt in range(iterations):
        # dot是矩阵乘法，* np.multiply()是点乘
        gradient = (np.repeat((X.dot(theta) - y), n_feature, axis=1) * X).sum(axis=0).reshape(n_feature, 1) / n_sample
        theta -= alpha * gradient
        J.append(cost_function(y, X, theta))
    return theta, J


def show_regression_line(data, theta):
    # 原始数据
    _x = data['Size'].tolist()
    _y = data['Bedrooms'].tolist()
    _z = data['Price'].tolist()
    # 回归模型
    x = np.linspace(data.Size.min(), data.Size.max(), 100)
    y = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)
    z = theta[0] + (theta[1] * x) + (theta[2] * y)

    fig = plt.figure(2, figsize=(12, 8))
    ax = Axes3D(fig)

    ax.plot(x, y, z, 'r')  # 回归模型
    ax.scatter(_x, _y, _z, label='Training Data')  # 原始数据
    ax.legend(loc=2)

    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    # plt.show()


def show_cost_descent(J):
    fig = plt.figure(3, figsize=(12, 8))
    ax = fig.add_subplot()
    ax.plot(J)
    # plt.show()

# def show_cost(svm=pd.DataFrame()):
#     n_sample = svm.shape[0]
#     n_feature = svm.shape[1] - 1
#     X = np.array(svm.iloc[:, :-1])
#     y = np.array(svm.iloc[:, -1]).reshape(n_sample, 1)


if __name__ == "__main__":
    pd_data = load_data(DATA_PATH, True)
    # pd_data = pre_process(pd_data)
    # print(pd_data)
    theta, J = regression(pd_data, alpha=0.01, iterations=1000)
    # print(theta)
    show_regression_line(pd_data, theta)
    show_cost_descent(J)
    plt.show()