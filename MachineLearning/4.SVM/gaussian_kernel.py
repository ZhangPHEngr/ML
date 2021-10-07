# -*- coding: utf-8 -*-
"""
@Project: MachineLearning
@File   : gaussian_kernel.py
@Author : Zhang P.H
@Date   : 2021/10/7
@Desc   :
"""
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio

# kernek function 高斯核函数
def gaussian_kernel(x1, x2, sigma):
    return np.exp(- np.power(x1 - x2, 2).sum() / (2 * (sigma ** 2)))

DATA = '../Data/svm/ex6data2.mat'


def load_data(Path, vis=True):
    mat = sio.loadmat(Path)
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    data['y'] = mat.get('y')
    if vis:
        sns.set(context="notebook", style="white", palette=sns.diverging_palette(240, 10, n=2))
        sns.lmplot('X1', 'X2', hue='y', data=data,
                   size=5,
                   fit_reg=False,
                   scatter_kws={"s": 10}
                   )

    return data

def svm_gaussian(data):
    svc = svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)
    svc.fit(data[['X1', 'X2']], data['y'])
    # svc.score(k_means[['X1', 'X2']], k_means['y'])
    predict_prob = svc.predict_proba(data[['X1', 'X2']])[:, 1]
    print(predict_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['X1'], data['X2'], s=30, c=predict_prob, cmap='Reds')

if __name__ == '__main__':
    data = load_data(DATA)
    svm_gaussian(data)
    plt.show()
