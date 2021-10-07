#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @FileName  :data_parser.py
# @Time      :2021/9/12 10:35
# @Author    :ZhangP.H
# Function Description:
import numpy as np
import pandas as pd


def load_data(data_path, names):
    data = pd.read_csv(data_path, header=None, names=names)
    return data


if __name__ == "__main__":
    pass
