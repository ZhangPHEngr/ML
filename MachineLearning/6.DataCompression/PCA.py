# -*- coding: utf-8 -*-
"""
@Project: MachineLearning
@File   : PCA.py
@Author : Zhang P.H
@Date   : 2021/10/7
@Desc   :
"""
"""
数据压缩：
    目的：减少数据维度，便于减少计算内存，加快学习算法，便于可视化展示
    方法：PCA(Principal Component Analysis) 主成分分析法
        简介：根据需求保留大权重特征向量对应的维度，舍去次要维度
        算法流程：
            1.数据归一化
            2.计算协方差矩阵
            3.分解协方差矩阵
            4.选取特征向量
            5.数据降维
        维度选择：根据特征值占比
        特征重建：数据降维的逆过程
        特点：线性回归要求最小预测误差(y方向)，PCA要求最小投影误差(投影)
             正则化并没有减少特征，PCA丢失了特征
"""

import numpy as np


def PCA(X, k):  # k is the components you want
    n_samples, n_features = X.shape  # 输出  行，列
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # 归一化
    norm_X = X - mean
    # 协方差计算
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)  # X.T*X
    # 特征分解
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]  # 输出特征值和特征向量对应的tuple
    print(eig_pairs)
    # 选取特征向量
    eig_pairs.sort(reverse=True)  # 默认按照第一个key排序
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # 数据降维
    data = np.dot(norm_X, np.transpose(feature))
    return data


if __name__ == '__main__':
    X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    print(PCA(X, 1))
