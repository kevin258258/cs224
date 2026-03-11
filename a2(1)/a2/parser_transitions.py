#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2024-2025：作业 2
parser_transitions.py：完成部分解析的算法。
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""

import sys

class PartialParse(object):
    def __init__(self, sentence):
        """初始化该部分解析对象。

        @param sentence (list of str): 待解析句子，以词列表表示。
                                       你的代码不应修改该句子。
        """
        # 保存原句仅用于记录用途。不要在你的代码里修改它。
        self.sentence = sentence

        self.stack = ["ROOT"]  # 栈初始化为包含根节点的列表
        self.buffer = sentence[:]  # 缓冲区初始化为输入句子的浅拷贝
        self.dependencies = []  # 依存关系初始化为空列表

        ### YOUR CODE HERE (3 Lines)
        ### 你的代码应初始化以下字段：
        ###     self.stack：当前栈，用列表表示，栈顶是列表最后一个元素。
        ###     self.buffer：当前缓冲区，用列表表示，缓冲区第一个元素在列表开头。
        ###     self.dependencies：目前产生的依存关系列表。每个元素是
        ###             形如 (head, dependent) 的二元组。
        ###             列表中元素顺序不重要。
        ###
        ### 注意：根节点 token 应使用字符串 "ROOT" 表示
        ### 注意：如果你需要用 sentence 初始化内容，务必不要直接引用
        ###       sentence 对象，也就是不要修改 sentence 对象本身。


        ### END YOUR CODE


    def parse_step(self, transition):
        """对该部分解析执行一步解析，方法是应用给定转移操作。

        @param transition (str): 取值为 "S"、"LA" 或 "RA" 的字符串，分别表示
                                 shift、left-arc、right-arc 转移。你可以假设
                                 传入的转移是合法的。
        """
        ### YOUR CODE HERE (~7-12 Lines)
        ### TODO：
        ###     按照 PDF 讲义中的描述，实现一次解析步骤，即下面三种操作：
        ###         1. Shift
        ###         2. Left Arc
        ###         3. Right Arc
        if transition == "S":
            self.stack.append(self.buffer.pop(0))
        elif transition == "LA":
            self.dependencies.append((self.stack[-1], self.stack[-2]))
            self.stack.pop(-2)
        elif transition == "RA":
            self.dependencies.append((self.stack[-2], self.stack[-1]))
            self.stack.pop()

        ### END YOUR CODE

    def parse(self, transitions):
        """将给定的转移序列应用到该 PartialParse。

        @param transitions (list of str): 按执行顺序排列的转移列表

        @return dependencies (list of string tuples): 解析句子得到的依存关系列表，
                                                       每个元素是形如 (head, dependent) 的元组。
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    """使用模型按小批次解析一组句子。

    @param sentences (list of list of str): 待解析句子列表
                                            （每个句子是词列表，每个词是字符串）
    @param model (ParserModel): 做解析决策的模型。假设其有函数
                                model.predict(partial_parses)，输入为 PartialParse 列表，
                                返回每个解析对象的下一步转移。也就是说调用
                                    transitions = model.predict(partial_parses)
                                后，transitions[i] 是应用到 partial_parses[i] 的下一步转移。
    @param batch_size (int): 每个小批次包含的 PartialParse 数量


    @return dependencies (list of dependency lists): 一个列表，其中每个元素是对应句子的
                                                     依存关系结果。顺序应与 `sentences` 一致，
                                                     即 `dependencies[i]` 对应 `sentences[i]`。
    """
    dependencies = []

    ### YOUR CODE HERE (~8-10 Lines)
    ### TODO：
    ###     实现 minibatch parse 算法。该算法的伪代码在 PDF 讲义中给出。
    ###
    ###     注意：浅拷贝（PDF 中有说明）可以用 Python 的 "=" 方式创建，例如
    ###                 unfinished_parses = partial_parses[:].
    ###             此时 `unfinished_parses` 是 `partial_parses` 的浅拷贝。
    ###             在 Python 中，浅拷贝列表（如 `unfinished_parses`）不会创建新对象实例，
    ###             两个列表会引用同一批对象。
    ###             在本题中，`partial_parses` 存的是部分解析对象，`unfinished_parses`
    ###             也只保存对这些同一对象的引用。因此，不要使用 `del` 从
    ###             `unfinished_parses` 中删除对象。这可能释放 `partial_parses`
    ###             仍在访问的底层内存，从而导致程序崩溃。



    ### END YOUR CODE
    partial_parses = [PartialParse(sentence) for sentence in sentences]
    unfinished_parses = partial_parses[:]

    while len(unfinished_parses) > 0:
        minibatch = unfinished_parses[:batch_size]
        transitions = model.predict(minibatch)

        for partial_parse, transition in zip(minibatch, transitions):
            partial_parse.parse_step(transition)

        unfinished_parses = [pp for pp in unfinished_parses
                             if len(pp.buffer) > 0 or len(pp.stack) > 1]

    dependencies = [pp.dependencies for pp in partial_parses]

    return dependencies


def test_step(name, transition, stack, buf, deps,
              ex_stack, ex_buf, ex_deps):
    """测试单步解析是否得到预期输出"""
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    assert stack == ex_stack, \
        "{:} 测试得到的栈为 {:}，期望为 {:}".format(name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} 测试得到的缓冲区为 {:}，期望为 {:}".format(name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} 测试得到的依存关系为 {:}，期望为 {:}".format(name, deps, ex_deps)
    print("{:} 测试通过！".format(name))


def test_parse_step():
    """`PartialParse.parse_step` 的简单测试
    警告：这些测试并不完备
    """
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
    """`PartialParse.parse` 的简单测试
    警告：这些测试并不完备
    """
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected,  \
        "parse 测试得到的依存关系为 {:}，期望为 {:}".format(dependencies, expected)
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse 测试失败：输入句子不应被修改"
    print("parse 测试通过！")


class DummyModel(object):
    """用于测试 `minibatch_parse` 的虚拟模型
    """
    def __init__(self, mode = "unidirectional"):
        self.mode = mode

    def predict(self, partial_parses):
        if self.mode == "unidirectional":
            return self.unidirectional_predict(partial_parses)
        elif self.mode == "interleave":
            return self.interleave_predict(partial_parses)
        else:
            raise NotImplementedError()

    def unidirectional_predict(self, partial_parses):
        """先把所有词移入栈中，然后若句首词为 "right" 则始终做右弧，
        否则始终做左弧。
        """
        return [("RA" if pp.stack[1] == "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

    def interleave_predict(self, partial_parses):
        """先把所有词移入栈中，然后交替执行右弧和左弧。
        """
        return [("RA" if len(pp.stack) % 2 == 0 else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

def test_dependencies(name, deps, ex_deps):
    """测试给定依存关系是否与期望一致"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} 测试得到的依存关系列表为 {:}，期望为 {:}".format(name, deps, ex_deps)


def test_minibatch_parse():
    """`minibatch_parse` 的简单测试
    警告：这些测试并不完备
    """

    # 单向弧测试
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))

    # 越界场景测试
    sentences = [["right"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0], (('ROOT', 'right'),))

    # 混合弧测试
    sentences = [["this", "is", "interleaving", "dependency", "test"]]
    deps = minibatch_parse(sentences, DummyModel(mode="interleave"), 1)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'is'), ('dependency', 'interleaving'),
                      ('dependency', 'test'), ('is', 'dependency'), ('is', 'this')))
    print("minibatch_parse 测试通过！")


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        raise Exception("你没有提供有效关键字。执行该脚本时请传入 'part_c' 或 'part_d'")
    elif args[1] == "part_c":
        test_parse_step()
        test_parse()
    elif args[1] == "part_d":
        test_minibatch_parse()
    else:
        raise Exception("你没有提供有效关键字。执行该脚本时请传入 'part_c' 或 'part_d'")
