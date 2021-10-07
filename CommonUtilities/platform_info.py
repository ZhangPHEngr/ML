#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @FileName  :platform_info.py
# @Time      :2021/9/4 17:13
# @Author    :ZhangP.H
# Function Description: 提供和运行平台(windows, linux)相关的帮助

import platform


def check_platform(platform_name=None):
    """
    查询当前程序运行平台，windows or linux
    :param platform_name: 平台名，如果不写就返回平台查询结果，写就做判断是否一致
    :return: windows linux True False
    """
    res = platform.system().lower()
    if not platform_name:
        return res
    elif platform_name == res:
        return True
    else:
        return False


if __name__ == "__main__":
    print(check_platform("windows"))
