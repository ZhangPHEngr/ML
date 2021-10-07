# -*- coding: utf-8 -*-
"""
@Project: MachineLearning
@File   : k_means.py
@Author : Zhang P.H
@Date   : 2021/10/7
@Desc   :
K-Means聚类算法：
    1. 随机初始化K个点作为聚类中心
    2. 遍历所有样本点，计算到K个聚类中心的距离，并归类为距离最近的类
    3. 分别对当前K类所有样本求均值得到新的聚类中心
    4. 判决新老聚类中心是否发生变动，任一变动就要重新执行2-4，否则完成聚类
另外：
    K的个数需要通过“肘部法则”选取
    K确定后要进行多次K-means聚类计算，防止因随机初始化导致的局部极小值
    距离计算方法除了欧氏距离还有余弦相似度等度量准则

更多方法：
https://scikit-learn.org.cn/view/108.html#2.3.2.%20K-means
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
import scipy.io as sio

DATA = '../Data/k_means/ex7data2.mat'


def load_data(path, vis=True):
    mat = sio.loadmat(path)
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    if vis:
        sns.set(context="notebook", style="white")
        sns.lmplot('X1', 'X2', data=data, fit_reg=False)
    return data


def k_means(data, k, vis=True):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    data['C'] = kmeans.labels_
    if vis:
        sns.lmplot('X1', 'X2', hue='C', data=data, fit_reg=False)


if __name__ == '__main__':
    data = load_data(DATA)
    k_means(data, 3)
    plt.show()
