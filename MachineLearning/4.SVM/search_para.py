# -*- coding: utf-8 -*-
"""
@Project: MachineLearning
@File   : search_para.py
@Author : Zhang P.H
@Date   : 2021/10/7
@Desc   :
1.使用两部分样本数据，一部分作为训练集，一部分作为交叉验证集
2.对候选的参数进行逐个带入训练，而后对训练结果进行交叉验证，得到置信度
3.选择置信度最高的模型参数作为最优参数
"""

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import pandas as pd
import scipy.io as sio

DATA = '../Data/svm/ex6data3.mat'


def load_data(path):
    mat = sio.loadmat(path)
    # training
    training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    training['y'] = mat.get('y')
    # cross validate
    cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
    cv['y'] = mat.get('yval')
    return training, cv


def manual_search(training, cv):
    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    combination = [(C, gamma) for C in candidate for gamma in candidate]
    search = []

    for C, gamma in combination:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(training[['X1', 'X2']], training['y'])
        search.append(svc.score(cv[['X1', 'X2']], cv['y']))  # 每次计算出一个置信度
    best_score = search[np.argmax(search)]
    best_param = combination[np.argmax(search)]
    print(best_score, best_param)

    best_svc = svm.SVC(C=100, gamma=0.3)
    best_svc.fit(training[['X1', 'X2']], training['y'])
    ypred = best_svc.predict(cv[['X1', 'X2']])

    print(metrics.classification_report(cv['y'], ypred))


def sklearn_search(training, cv):
    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

    svc = svm.SVC()
    parameters = {'C': candidate, 'gamma': candidate}
    clf = GridSearchCV(svc, parameters, n_jobs=-1)
    clf.fit(training[['X1', 'X2']], training['y'])
    print(clf.best_params_)

    ypred = clf.predict(cv[['X1', 'X2']])
    print(metrics.classification_report(cv['y'], ypred))


if __name__ == '__main__':
    training, cv = load_data(DATA)
    manual_search(training, cv)
    sklearn_search(training, cv)
