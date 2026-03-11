#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2021-2022：作业 3
general_utils.py：通用工具函数。
Sahil Chopra <schopra8@stanford.edu>
"""

import numpy as np


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    按小批次遍历给定数据。你可以像下面这样使用本函数进行小批次迭代：

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    或者用于多个数据源：

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    参数：
        data: 两种可能形式：
            - list 或 numpy array
            - list，其中每个元素是 list 或 numpy array
        minibatch_size: 单个小批次允许的最大样本数
        shuffle: 是否打乱返回数据的顺序
    返回：
        minibatches: 返回值取决于 `data` 形式：
            - 如果 data 是 list/array，则逐次返回下一个小批次数据。
            - 如果 data 是由 list/array 构成的 list，则逐次返回每个元素的下一批数据。
              这可用于同时迭代多个数据源（例如特征与标签）。

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [_minibatch(d, minibatch_indices) for d in data] if list_data \
            else _minibatch(data, minibatch_indices)


def _minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def test_all_close(name, actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("{:} 失败，期望输出形状为 {:}，实际为 {:}"
                         .format(name, expected.shape, actual.shape))
    if np.amax(np.fabs(actual - expected)) > 1e-6:
        raise ValueError("{:} 失败，期望为 {:}，实际值为 {:}".format(name, expected, actual))
    else:
        print(name, "通过！")
