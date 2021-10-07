# -*- coding: utf-8 -*-
"""
@Project: MachineLearning
@File   : linear_svm.py
@Author : Zhang P.H
@Date   : 2021/10/7
@Desc   :
"""
"""
sklearn库简介: 
https://www.cnblogs.com/cafe3165/p/9145427.html
https://zhuanlan.zhihu.com/p/33420189
"""
import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt

DATA = '../Data/svm/ex6data1.mat'


def load_data(Path, vis=True):
    mat = sio.loadmat(Path)
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    data['y'] = mat.get('y')
    if vis:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data['X1'], data['X2'], s=50, c=data['y'], cmap='Reds')
        ax.set_title('Raw k_means')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')

    return data


def svm_linear(data, vis=True):
    svc1 = sklearn.svm.LinearSVC(C=100, loss='hinge')  # 设置svm基本参数
    svc1.fit(data[['X1', 'X2']], data['y'])
    svc1.score(data[['X1', 'X2']], data['y'])

    data['SVM1 Confidence'] = svc1.decision_function(data[['X1', 'X2']])

    predict_res = pd.Series(svc1.predict(data[['X1', 'X2']]), name="predict_res")
    res = pd.concat([data['SVM1 Confidence'], predict_res], axis=1)
    print(res)

    if vis:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='RdBu')
        ax.set_title('SVM (C=1) Decision Confidence')
    print(data.head())


if __name__ == '__main__':
    data_sample = load_data(DATA)
    svm_linear(data_sample)
    plt.show()

