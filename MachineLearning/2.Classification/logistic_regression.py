#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @FileName  :logistic_regression.py
# @Time      :2021/9/12 10:38
# @Author    :ZhangP.H
# Function Description:
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import scipy.optimize as opt

"""
分类问题，其实是一个逻辑回归问题，也即对分类数据进行回归（引入sigmoid函数，使用非线性回归，而不是之前的线性回归）得到回归函数，以0.5为概率分界线判决
举例，如下程序一个二维分类问题，其实是一个三维逻辑回归问题，回归函数与z=0.5平面的交线就是二维分类线
使用sigmoid的原因是，可以将回归范围限制在0-1之间，以完成二分类，也与概率区间对应好，更深层是因为最大熵原理
"""


DATA_PATH = '../Data/ex2data1.txt'


def load_data(data_path, do_vis=False):
    data = pd.read_csv(data_path, header=None, names=['exam1', 'exam2', 'admission'])
    if do_vis:
        sns.jointplot(x='exam1', y='exam2', hue='admission', data=data)
        plt.show()  # 看下数据的样子
    # 数据预处理
    # Data = pre_process(Data)
    data.insert(0, 'Ones', 1)  # 增加一列1便于后序特征计算  y = a_0 * x_0 + a_1 * x_1
    return data


def sigmoid(z):
    """
    sigmoid函数映射
    :param z: 实数 或 ndarray
    :return: 实数 或 ndarray
    """
    return 1 / (1 + np.exp(-z))


def hypothesis_model(x, theta):
    """
    假设的模型，根据输入x和模型参数theta返回结果
    :param x: ndarray  (1 + n_feature) * 1
    :param theta: ndarray  (1 + n_feature) * 1
    :return: 实数，模型映射结果,
    """
    return sigmoid(x.T @ theta)


def cost_function(theta, x, y):
    """
    误差函数计算
    :param x: ndarray  m * (1 + n_feature)
    :param y: ndarray  m * 1
    :param theta: ndarray  (1 + n_feature) * 1
    :return: 实数
    """
    h = sigmoid(x @ theta)
    return np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))


def gradient(theta, x, y):
    """
    theta确定时的梯度
    :param x: ndarray  m * (1 + n_feature)
    :param y: ndarray  m * 1
    :param theta: ndarray  (1 + n_feature) * 1
    :return: (1 + n_feature) * 1 梯度结果
    """
    return x.T @ (sigmoid(x @ theta) - y) / x.shape[0]


def classification(data=pd.DataFrame()):
    x = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])
    theta = np.array([[0., 0., 0.]])
    res = opt.minimize(fun=cost_function, x0=theta, args=(x, y), method='Newton-CG', jac=gradient)
    return res


def show_2D(data, theta):
    # 回归模型
    x = np.linspace(data.exam1.min(), data.exam1.max(), 100)
    y = -(theta[0]/theta[2] + theta[1]/theta[2]*x)

    # fig, ax = plt.subplots(figsize=(12, 8))
    ax = sns.jointplot(x='exam1', y='exam2', hue='admission', data=data).ax_joint
    ax.plot(x, y, 'r')  # 回归模型

    ax.set_xlabel('exam1')
    ax.set_ylabel('exam2')
    ax.set_title('2.Classification')
    # plt.show()


def show_3D(data, theta):
    fig = plt.figure(2, figsize=(12, 8))
    ax = Axes3D(fig)
    ax.scatter(data.exam1, data.exam2, data.admission)

    # 回归模型
    x = np.linspace(data.exam1.min(), data.exam1.max(), 100)
    y = np.linspace(data.exam2.min(), data.exam2.max(), 100)
    X, Y = np.meshgrid(x, y)
    z = theta[0] + (theta[1] * X) + (theta[2] * Y)
    f = sigmoid(z)
    ax.plot_surface(X, Y, f, color='r', alpha=0.6)
    ax.plot_surface(X, Y, Z=X * 0 + 0.5, color='g', alpha=0.6)
    ax.set_xlabel('exam1')
    ax.set_ylabel('exam2')
    ax.set_title('2.Classification')


def show_cost_descent(J):
    fig = plt.figure(1, figsize=(12, 8))
    ax = fig.add_subplot()
    ax.plot(J, 'r')
    # plt.show()


if __name__ == "__main__":
    pd_data = load_data(DATA_PATH, False)
    res = classification(pd_data)
    # show_classification_line(pd_data, res.x)
    show_3D(pd_data, res.x)
    show_2D(pd_data, res.x)
    # show_cost_descent(J)
    plt.show()

    # x = np.array(pd_data.iloc[:, :-1])
    # y = np.array(pd_data.iloc[:, -1]).reshape(pd_data.index.size, 1)
    # theta = np.zeros((x.shape[1], 1))
    # print(cost_function(x, y, theta))
    # print(res)

