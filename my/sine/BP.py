# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:39:55 2018
@author: Administrator
"""

import math
import random
import numpy
import matplotlib.pyplot as plt

random.seed(0)  # 使random函数每一次生成的随机数值都相等


def rand(a, b):
    return (b - a) * random.random() + a  # 生成a到b的随机数


def make_matrix(m, n, fill=0.0):  # 创建一个指定大小的矩阵
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


# 定义tanh函数和它的导数
def tanh(x):
    return numpy.tanh(x)


def tanh_derivate(x):
    return 1 - numpy.tanh(x) * numpy.tanh(x)  # tanh函数的导数


class BPNeuralNetwork:
    def __init__(self):  # 初始化变量
        self.input_n = 0
        self.hidden1_n = 0
        self.hidden2_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden1_cells = []
        self.hidden2_cells = []
        self.output_cells = []
        self.input_weights = []
        self.hidden_weights = []
        self.output_weights = []
        self.input_correction = []
        self.hidden_correction = []
        self.output_correction = []
        # 三个列表维护：输入层，隐含层，输出层神经元

    def setup(self, ni, nh1, nh2, no):
        self.input_n = ni + 1  # 输入层+偏置项
        self.hidden1_n = nh1 + 1  # 隐含层 1
        self.hidden2_n = nh2 + 1  # 隐含层 2
        self.output_n = no  # 输出层

        # 初始化神经元
        self.input_cells = [1.0] * self.input_n
        self.hidden1_cells = [1.0] * self.hidden1_n
        self.hidden2_cells = [1.0] * self.hidden2_n
        self.output_cells = [1.0] * self.output_n

        # 初始化连接边的边权
        self.input_weights = make_matrix(self.input_n, self.hidden1_n)  # 邻接矩阵存边权：输入层->隐藏层
        self.hidden_weights = make_matrix(self.hidden1_n, self.hidden2_n)
        self.output_weights = make_matrix(self.hidden2_n, self.output_n)  # 邻接矩阵存边权：隐藏层->输出层
        # 初始化bias
        self.h1_b = make_matrix(self.hidden1_n, 1)
        self.h2_b = make_matrix(self.hidden2_n, 1)
        self.o_b = make_matrix(self.output_n, 1)
        # 随机初始化边权：为了反向传导做准备--->随机初始化的目的是使对称失效
        for i in range(self.input_n):
            for h in range(self.hidden1_n):
                self.input_weights[i][h] = rand(-1, 1)  # 由输入层第i个元素到隐藏层1第j个元素的边权为随机值
        for i in range(self.hidden1_n):
            for h in range(self.hidden2_n):
                self.hidden_weights[i][h] = rand(-1, 1)  # 由隐藏层1第i个元素到隐藏层2第j个元素的边权为随机值
        for h in range(self.hidden2_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-1, 1)  # 由隐藏层2第i个元素到输出层第j个元素的边权为随机值
        # 随机初始化bias
        for i in range(self.hidden1_n):
            self.h1_b[i] = rand(-1, 1)
        for i in range(self.hidden2_n):
            self.h2_b[i] = rand(-1, 1)
        for i in range(self.output_n):
            self.o_b[i] = rand(-1, 1)
        # 保存校正矩阵，为了以后误差做调整
        self.input_correction = make_matrix(self.input_n, self.hidden1_n)
        self.hidden_correction = make_matrix(self.hidden1_n, self.hidden2_n)
        self.output_correction = make_matrix(self.hidden2_n, self.output_n)

        # 输出预测值

    def predict(self, inputs):
        # 对输入层进行操作转化样本
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]  # n个样本从0~n-1
        # 计算隐藏层的输出，每个节点最终的输出值就是权值*节点值的加权和
        for j in range(self.hidden1_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
                # 此处为何是先i再j，以隐含层节点做大循环，输入样本为小循环，是为了每一个隐藏节点计算一个输出值，传输到下一层
            self.hidden1_cells[j] = tanh(total - self.h1_b[j])  # 此节点的输出是前一层所有输入点和到该点之间的权值加权和
        for m in range(self.hidden2_n):
            total = 0.0
            for i in range(self.hidden1_n):
                total += self.hidden1_cells[i] * self.hidden_weights[i][m]

            self.hidden2_cells[m] = tanh(total - self.h2_b[m])  # 此节点的输出是前一层所有输入点和到该点之间的权值加权和
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden2_n):
                total += self.hidden2_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = tanh(total - self.o_b[k])  # 获取输出层每个元素的值
        return self.output_cells[:]  # 最后输出层的结果返回

    # 反向传播算法
    def back_propagate(self, case, label, learn, correct):
        self.predict(case)  # 对实例进行预测
        output_deltas = [0.0] * self.output_n  # 初始化矩阵
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]  # 正确结果和预测结果的误差：0,1，-1
            output_deltas[o] = tanh_derivate(self.output_cells[o]) * error  # 误差稳定在0~1内
        # 隐含层误差
        hidden1_deltas = [0.0] * self.hidden1_n
        hidden2_deltas = [0.0] * self.hidden2_n
        for h in range(self.hidden2_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden2_deltas[h] = tanh_derivate(self.hidden2_cells[h]) * error
            # 反向传播算法求W
        # 更新隐藏层->输出权重
        for h2 in range(self.hidden2_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden2_cells[h2]
                # 调整权重：上一层每个节点的权重学习*变化+矫正率
                self.output_weights[h2][o] += learn * change + correct * self.output_correction[h2][o]
                self.output_correction[h2][o] = change
        # 更新隐藏1层->隐藏2权重
        for h1 in range(self.hidden1_n):
            for o in range(self.hidden2_n):
                change = hidden2_deltas[o] * self.hidden1_cells[h1]
                # 调整权重：上一层每个节点的权重学习*变化+矫正率
                self.hidden_weights[h1][o] += learn * change + correct * self.hidden_correction[h1][o]
                self.hidden_correction[h1][o] = change
        # 更新输入->隐藏层的权重
        for i in range(self.input_n):
            for h in range(self.hidden1_n):
                change = hidden1_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # 更新bias
        for o in range(self.output_n):
            self.o_b[o] = self.o_b[o] - learn * output_deltas[o]
        for h2 in range(self.hidden2_n):
            self.h2_b[h2] = self.h2_b[h2] - learn * hidden2_deltas[h2]
        for h1 in range(self.hidden1_n):
            self.h1_b[h1] = self.h1_b[h1] - learn * hidden1_deltas[h1]

        error = 0.0
        for o in range(len(label)):
            error = 0.5 * (label[o] - self.output_cells[o]) ** 2  # 平方误差函数
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        for i in range(limit):  # 设置迭代次数
            error = 0.0
            for j in range(len(cases)):  # 对输入层进行访问
                label = labels[j]
                case = cases[j]
                error += self.back_propagate(case, label, learn, correct)  # 样例，标签，学习率，正确阈值

    def test(self):  # 学习正弦函数
        cases = []  # 测试样
        for i in range(0, 21, 1):
            cases.append([i * math.pi / 10])
            labels = numpy.sin(cases)
        self.setup(1, 10, 10, 1)  # 初始化神经网络：输入层，隐藏层，输出层元素个数
        self.train(cases, labels, 10000, 0.05, 0.1)
        test = []  # 训练范围外的数据
        yables = []

        for i in range(0, 201, 1):
            test.append([i * math.pi / 100])
        for case in test:
            yables.append(self.predict(case))

        x = numpy.arange(0.0, 2.0, 0.01)
        plt.figure()
        l1, = plt.plot(x * math.pi, numpy.sin(x * math.pi), color='red')
        l2, = plt.plot(test, yables, color='green')
        plt.legend(handles=[l1, l2, ], labels=['original', 'test predict'], loc='best')
        plt.xticks([0, numpy.pi / 2, numpy.pi, 3 * numpy.pi / 2, 2 * numpy.pi],
                   [r'$0$', r'$+\pi/2$', r'$+\pi$', r'$+\pi*3/2$', r'$+\pi*2$'])
        plt.show()

        zables = []
        for a in cases:
            zables.append(self.predict(a))
        plt.figure()
        l3, = plt.plot(x * math.pi, numpy.sin(x * math.pi), color='red')
        l4, = plt.plot(cases, zables, color='green')
        plt.legend(handles=[l3, l4, ], labels=['original', 'train predict'], loc='best')
        plt.xticks([0, numpy.pi / 2, numpy.pi, 3 * numpy.pi / 2, 2 * numpy.pi],
                   [r'$0$', r'$+\pi/2$', r'$+\pi$', r'$+\pi*3/2$', r'$+\pi*2$'])
        plt.show()


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()