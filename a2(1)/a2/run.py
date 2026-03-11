#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2023-2024：作业 2
run.py：运行依存句法分析器。
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""
from datetime import datetime
import os
import pickle
import math
import time
import argparse

from torch import nn, optim
import torch
from tqdm import tqdm

from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter

parser = argparse.ArgumentParser(description='在 PyTorch 中训练神经依存句法分析器')
parser.add_argument('-d', '--debug', action='store_true', help='是否进入调试模式')
args = parser.parse_args()

# -----------------
# 核心函数
# -----------------
def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
    """训练神经依存句法分析器。

    @param parser (Parser): 神经依存句法分析器
    @param train_data ():
    @param dev_data ():
    @param output_path (str): 保存模型权重与结果的路径。
    @param batch_size (int): 每个批次中的样本数
    @param n_epochs (int): 训练轮数
    @param lr (float): 学习率
    """
    best_dev_UAS = 0


    ### YOUR CODE HERE (~2-7 lines)
    ### TODO：
    ###      1) 在变量 `optimizer` 中构建 Adam 优化器
    ###      2) 在变量 `loss_func` 中构建交叉熵损失函数，使用 `mean`
    ###         归约方式（默认值）
    ###
    ### 提示：使用 `parser.model.parameters()` 向优化器传入
    ###       需要更新的参数。
    ### 可参考以下文档：
    ###     Adam 优化器：https://pytorch.org/docs/stable/optim.html
    ###     交叉熵损失：https://pytorch.org/docs/stable/nn.html#crossentropyloss
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    ### END YOUR CODE

    for epoch in range(n_epochs):
        print("第 {:} / {:} 轮".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("验证集 UAS 达到新最佳，正在保存模型。")
            torch.save(parser.model.state_dict(), output_path)
        print("")


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    """训练神经依存句法分析器的单个 epoch。

    说明：在 PyTorch 中，可以通过声明模型当前处于训练模式 `model.train()`
    或评估模式 `model.eval()`，来自动启用或关闭 Dropout 层。

    @param parser (Parser): 神经依存句法分析器
    @param train_data ():
    @param dev_data ():
    @param optimizer (nn.Optimizer): Adam 优化器
    @param loss_func (nn.CrossEntropyLoss): 交叉熵损失函数
    @param batch_size (int): 批大小

    @return dev_UAS (float): 开发集上的无标签依存准确率（UAS）
    """
    parser.model.train() # 将模型置为训练模式，即启用 dropout 层
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()   # 清除优化器中累积的梯度
            loss = 0. # 在这里存储当前批次的损失
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()

            ### YOUR CODE HERE (~4-10 lines)
            ### TODO：
            ###      1) 将 train_x 送入模型前向传播，得到 `logits`
            ###      2) 使用参数 `loss_func` 计算 PyTorch 的 CrossEntropyLoss
            ###         输入为 `logits` 和 `train_y`，输出为 softmax(`logits`) 与 `train_y`
            ###         之间的交叉熵损失。注意 softmax(`logits`) 就是预测值（PDF 中的 y^）。
            ###      3) 反向传播损失
            ###      4) 调用优化器执行一步更新
            ### 可参考以下文档：
            ###     Optimizer Step：https://pytorch.org/docs/stable/optim.html#optimizer-step
            logits = parser.model(train_x)
            loss = loss_func(logits, train_y)
            loss.backward()
            optimizer.step()

            ### END YOUR CODE
            prog.update(1)
            loss_meter.update(loss.item())

    print ("训练集平均损失：{}".format(loss_meter.avg))

    print("正在开发集上评估",)
    parser.model.eval() # 将模型置为评估模式，即关闭 dropout 层
    dev_UAS, _ = parser.parse(dev_data)
    print("- 开发集 UAS：{:.2f}".format(dev_UAS * 100.0))
    return dev_UAS


if __name__ == "__main__":
    debug = args.debug

    assert (torch.__version__.split(".") >= ["1", "0", "0"]), "请安装 torch >= 1.0.0"

    print(80 * "=")
    print("初始化")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()
    model = ParserModel(embeddings)
    parser.model = model
    print("耗时 {:.2f} 秒\n".format(time.time() - start))

    print(80 * "=")
    print("训练")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005)

    if not debug:
        print(80 * "=")
        print("测试")
        print(80 * "=")
        print("恢复在开发集上效果最好的模型权重")
        parser.model.load_state_dict(torch.load(output_path))
        print("在测试集上进行最终评估",)
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
        print("- 测试集 UAS：{:.2f}".format(UAS * 100.0))
        print("完成！")
