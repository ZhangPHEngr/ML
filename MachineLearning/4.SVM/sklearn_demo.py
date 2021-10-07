# -*- coding: utf-8 -*-
"""
@Project: MachineLearning
@File   : sklearn_demo.py
@Author : Zhang P.H
@Date   : 2021/10/7
@Desc   : 用于学习sklearn库的使用
    英文文档：https://scikit-learn.org/stable/auto_examples/index.html#support-vector-machines
    中文文档：https://scikit-learn.org.cn/
    例程：https://www.cnblogs.com/luyaoblog/p/6775342.html
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


def binary_classification():
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC()
    clf.fit(X, y)
    res = clf.predict([[3., 3.]])
    print(res)


def show_divide():
    # 我们创建40个用来分割的数据点
    X, y = make_blobs(n_samples=40, centers=2, random_state=6)  # X:m*n y n
    print(X, y)
    # 拟合模型，并且为了展示作用，并不进行标准化
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # 绘制decision function的结果
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # 创造网格来评估模型
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    # 绘制决策边界和边际
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # 绘制支持向量
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


if __name__ == '__main__':
    show_divide()
