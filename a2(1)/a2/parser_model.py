#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2024-2025：作业 2
parser_model.py：用于依存句法分析的前馈神经网络
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParserModel(nn.Module):
    """带有嵌入层和两层隐藏层的前馈神经网络。
    ParserModel 会预测在给定部分解析配置下应执行的转移操作。

    PyTorch 说明：
        - `ParserModel` 是 `nn.Module` 的子类。在 PyTorch 中所有神经网络
          都是 `nn.Module` 的子类。
        - `__init__` 方法用于定义所有层和参数
          （嵌入层、线性层、dropout 层等）。
        - 当你创建类实例时（例如 `m = ParserModel()`），`__init__`
          会被自动调用。
        - `ParserModel` 的其他方法可访问带 `self.` 前缀的变量。因此，
          你希望在其他方法中使用的层、数值等都应加上 `self.` 前缀。
        - 更多 `nn.Module` 文档见：https://pytorch.org/docs/stable/nn.html
    """
    def __init__(self, embeddings, n_features=36,
        hidden_size=200, n_classes=3, dropout_prob=0.5):
        """初始化解析模型。

        @param embeddings (ndarray): 词向量矩阵 (num_words, embedding_size)
        @param n_features (int): 输入特征数量
        @param hidden_size (int): 隐藏单元数量
        @param n_classes (int): 输出类别数量
        @param dropout_prob (float): dropout 概率
        """
        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.embeddings = nn.Parameter(torch.tensor(embeddings))

        ### YOUR CODE HERE (~9-10 Lines)
        ### TODO：
        ###     1) 将 `self.embed_to_hidden_weight` 和 `self.embed_to_hidden_bias`
        ###        声明为 `nn.Parameter`。权重使用 `nn.init.xavier_uniform_` 初始化，
        ###        偏置使用默认参数的 `nn.init.uniform_` 初始化。
        self.embed_to_hidden_weight = nn.Parameter(torch.empty(self.n_features * self.embed_size, self.hidden_size))
        self.embed_to_hidden_bias = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.xavier_uniform_(self.embed_to_hidden_weight)
        nn.init.uniform_(self.embed_to_hidden_bias)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.hidden_to_logits_weight = nn.Parameter(torch.empty(self.hidden_size, self.n_classes))
        self.hidden_to_logits_bias = nn.Parameter(torch.empty(self.n_classes))
        nn.init.xavier_uniform_(self.hidden_to_logits_weight)
        nn.init.uniform_(self.hidden_to_logits_bias)
        ###     2) 构建 `self.dropout` 层。
        ###     3) 将 `self.hidden_to_logits_weight` 和 `self.hidden_to_logits_bias`
        ###        声明为 `nn.Parameter`。权重使用 `nn.init.xavier_uniform_` 初始化，
        ###        偏置使用默认参数的 `nn.init.uniform_` 初始化。
        ###
        ### 注意：可训练变量应声明为 `nn.Parameter`。这是把张量加入计算图并
        ###       支持按梯度更新的常见 API。
        ###       这里对权重使用 Xavier Uniform 初始化。经验上它通常比随机均匀初始化
        ###       更适合作为网络训练初值。
        ###       更多说明可参考：
        ###             http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        ###
        ### 可参考以下文档：
        ###     nn.Parameter：https://pytorch.org/docs/stable/nn.html#parameters
        ###     初始化：https://pytorch.org/docs/stable/nn.init.html
        ###     Dropout：https://pytorch.org/docs/stable/nn.html#dropout-layers
        ### 
        ### 提示见 PDF 讲义。




        ### END YOUR CODE

    def embedding_lookup(self, w):
        """利用 `w` 从嵌入矩阵 `self.embeddings` 中取出对应向量。
            @param w (Tensor): 词索引输入张量 (batch_size, n_features)

            @return x (Tensor): `w` 中词对应的嵌入张量
                                (batch_size, n_features * embed_size)
        """

        ### YOUR CODE HERE (~1-4 Lines)
        ### TODO：
        ###     1) 对 `w` 中每个索引 `i`，从 self.embeddings 中选取第 `i` 个向量
        ###     2) 如有必要，使用 `view` 对张量重塑形状
        x = self.embeddings[w]
        x = x.view(x.shape[0], -1)
        ###
        ### 注意：所有嵌入向量按矩阵形式堆叠存储。模型接收代表词序列的索引列表，
        ###       再通过该 lookup 函数将索引映射为嵌入序列。
        ###
        ###       这道题旨在考察你对 embedding lookup 的理解，
        ###       因此不要使用 `nn.Embedding` 这类高级 API
        ###       （因为本题就是要你实现它）。请关注张量形状，
        ###       必要时进行 reshape。运行前要明确每个张量的维度。
        ###
        ### PyTorch 提供了一些可用 API，本题中可任选其一（`nn.Embedding` 除外）：
        ###     Index select：https://pytorch.org/docs/stable/torch.html#torch.index_select
        ###     Gather：https://pytorch.org/docs/stable/torch.html#torch.gather
        ###     View：https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Flatten：https://pytorch.org/docs/stable/generated/torch.flatten.html


        ### END YOUR CODE
        return x


    def forward(self, w):
        """执行模型前向计算。

            注意：这里不会显式应用 softmax，因为损失函数 `nn.CrossEntropyLoss`
            内部已包含该步骤。
        

            PyTorch 说明：
                - 每个 `nn.Module` 对象（PyTorch 模型）都有 `forward` 函数。
                - 当你将 `nn.Module` 应用于输入张量 `w` 时，会自动调用该函数。
                  例如，若创建了 ParserModel 实例并这样调用：
                        model = ParserModel()
                        output = model(w) # 会调用 forward
                  则 `forward` 会作用于 `w`，结果保存在 `output`。
                - 更多说明见：https://pytorch.org/docs/stable/nn.html#torch.nn.Module.forward

        @param w (Tensor): token 输入张量 (batch_size, n_features)

        @return logits (Tensor): 预测张量（网络各层输出），
                                 未应用 softmax (batch_size, n_classes)
        """
        ### YOUR CODE HERE (~3-5 lines)
        ### TODO：
        x = self.embedding_lookup(w)
        hidden = F.relu(x @ self.embed_to_hidden_weight + self.embed_to_hidden_bias)
        hidden = self.dropout(hidden)
        logits = hidden @ self.hidden_to_logits_weight + self.hidden_to_logits_bias

        ###     按写作说明完成前向计算。此外，在 ReLU 后加入 `__init__`
        ###     中声明的 dropout 层。
        ###
        ### 注意：这里不对 logits 显式做 softmax，因为
        ### 损失函数（torch.nn.CrossEntropyLoss）会更高效地处理它。
        ###
        ### 可参考以下文档：
        ###     矩阵乘法：https://pytorch.org/docs/stable/torch.html#torch.matmul
        ###     ReLU：https://pytorch.org/docs/stable/nn.html?highlight=relu#torch.nn.functional.relu

        ### END YOUR CODE
        return logits


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='parser_model.py 的简单健全性检查')
    parser.add_argument('-e', '--embedding', action='store_true', help='检查 embeding_lookup 函数')
    parser.add_argument('-f', '--forward', action='store_true', help='检查 forward 函数')
    args = parser.parse_args()

    embeddings = np.zeros((100, 30), dtype=np.float32)
    model = ParserModel(embeddings)

    def check_embedding():
        inds = torch.randint(0, 100, (4, 36), dtype=torch.long)
        selected = model.embedding_lookup(inds)
        assert np.all(selected.data.numpy() == 0), "embedding lookup 结果：" \
                                      + repr(selected) + " 含有非零元素。"

    def check_forward():
        inputs =torch.randint(0, 100, (4, 36), dtype=torch.long)
        out = model(inputs)
        expected_out_shape = (4, 3)
        assert out.shape == expected_out_shape, "forward 输出形状为：" + repr(out.shape) + \
                                                "，与期望值 " + repr(expected_out_shape) + " 不一致"

    if args.embedding:
        check_embedding()
        print("Embedding_lookup 检查通过！")

    if args.forward:
        check_forward()
        print("Forward 检查通过！")
