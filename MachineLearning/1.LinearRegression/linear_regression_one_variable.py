#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @FileName  :linear_regression_one_variable.py
# @Time      :2021/9/5 17:23
# @Author    :ZhangP.H
# Function Description:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = '../Data/ex1data1.txt'

'''
线性回归流程：
    1. 设计多项式回归模型H：
    2. 设计误差函数J:
    3. 对误差函数各参数求导dJ:
    4. 迭代计算
单变量线性回归，自变量X只有一个维度，因变量y也是一个维度，求其满足的模型及参数

'''


def load_data(data_path, do_vis=False):
    data = pd.read_csv(data_path, header=None, names=['Population', 'Profit'])
    data.insert(0, 'Ones', 1)  # 增加一列1便于后序特征计算  y = a_0 * x_0 + a_1 * x_1
    if do_vis:
        data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
        plt.show()
    return data


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


def sig_val_regression(data=pd.DataFrame(), alpha=0.01, iterations=1000):
    """
    单变量线性回归
    :param data: pd m*(n + 1) m行个样本 n个维度的自变量（已经补充第一列为1便于计算） 最后一列为真值
    :param alpha: 学习率
    :param iterations: 迭代次数
    :return:
    """
    m = data.shape[0]
    n_feature = data.shape[1] - 1
    X = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1]).reshape(m, 1)
    theta = np.zeros((n_feature, 1))
    J = list()
    for cnt in range(iterations):
        # dot是矩阵乘法，* np.multiply()是点乘
        gradient = (np.repeat((X.dot(theta) - y), n_feature, axis=1) * X).sum(axis=0).reshape(n_feature, 1) / m
        theta -= alpha * gradient
        J.append(cost_function(y, X, theta))
    return theta, J


def show_regression_line(data, theta):
    # 回归模型
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = theta[0] + (theta[1] * x)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')  # 回归模型
    ax.scatter(data.Population, data.Profit, label='Training Data')  # 原始数据
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    # plt.show()
    # 原始数据以及拟合的直线


def show_cost(J):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(J)



if __name__ == "__main__":
    pd_data = load_data(DATA_PATH, False)
    theta, J = sig_val_regression(pd_data, alpha=0.01, iterations=1000)
    show_regression_line(pd_data, theta)
    show_cost(J)
    plt.show()
