import torch
from torch import nn
from torch.nn import init
import numpy as np
from tools import pytorch_tool as tool
###########################################################################################################
class INIT(object):
    def __init__(self):
        self.batch_size = 256  # 训练批次大小
        self.train_iter, self.test_iter = tool.load_data_fashion_mnist(self.batch_size)

        self.input_length = 28  # 输入图片的长
        self.input_width = 28  # 输入图片的宽
        self.num_outputs = 10  # 输出结果类别
        self.num_epochs = 5  # 训练次数
        self.lr = 0.1  # 学习率
###########################################################################################################
#从零实现
class ZERO(INIT):
    def __init__(self):
        super(ZERO,self).__init__()

    # 实现softmax运算
    def softmax(self,logits):
        return logits.exp() / logits.exp().sum(dim=1, keepdim=True)  # 这里应用了广播机制

    # 定义模型
    def net(self,X):
        return self.softmax(torch.mm(X.view((-1, self.input_length * self.input_width)), self.W) + self.b)

    # 定义损失函数
    def cross_entropy(self,y_hat, y):
        return - torch.log(y_hat.gather(1, y.view(-1, 1)))

    # 计算分类准确率
    def accuracy(self,y_hat, y):
        return (y_hat.argmax(dim=1) == y).float().mean().item()

    # 模型评估
    def evaluate_accuracy(self,data_iter, net):
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n

    def output(self):
        # 初始化模型参数
        num_inputs = self.input_length * self.input_width
        self.W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, self.num_outputs)), dtype=torch.float)
        self.b = torch.zeros(self.num_outputs, dtype=torch.float)
        self.W.requires_grad_(requires_grad=True)
        self.b.requires_grad_(requires_grad=True)
        #模型训练
        tool.train_model(self.net,self.evaluate_accuracy, self.train_iter, self.test_iter, self.cross_entropy, self.num_epochs, self.batch_size, [self.W, self.b], self.lr)

        # 结果预测
        tool.predict_model(self.test_iter, self.net)
###########################################################################################################
#简洁实现
from collections import OrderedDict
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)
class CONCISE(INIT):
    def __init__(self):
        super(CONCISE,self).__init__()

    # 模型评估
    def evaluate_accuracy(self, data_iter, net):
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n

    def output(self):
        # 定义和初始化模型
        model= nn.Sequential(
            OrderedDict([
                ('flatten', FlattenLayer()),
                ('linear', nn.Linear(self.input_length*self.input_width, self.num_outputs))])
            )
        init.normal_(model.linear.weight, mean=0, std=0.01)
        init.constant_(model.linear.bias, val=0)
        # softmax和交叉熵损失函数
        LOSS =nn.CrossEntropyLoss()
        # 定义优化算法
        OPTIMIZER = torch.optim.SGD(model.parameters(), lr=self.lr)
        # 模型训练
        tool.train_model(model, self.evaluate_accuracy, self.train_iter, self.test_iter, LOSS,
                         self.num_epochs, self.batch_size, None, None,OPTIMIZER)

        # 结果预测
        tool.predict_model(self.test_iter, model)
if __name__=='__main__':
    #ZERO().output()  # 输出从零实现的Softmax回归模型训练结果
    CONCISE().output()  # 输出简洁实现的Softmax回归模型训练结果
