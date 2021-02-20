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
        self.num_inputs = self.input_width * self.input_length
        self.num_outputs = 10  # 输出结果类别
        self.num_epochs = 5  # 训练次数
        self.lr = 0.5  # 学习率

        self.num_hiddens = 256  # 隐藏层神经元
###########################################################################################################
#从零实现
class ZERO(INIT):
    def __init__(self):
        super(ZERO,self).__init__()

    # 定义激活函数
    def relu(self,X):
        return torch.max(input=X, other=torch.tensor(0.0))

    # 定义模型
    def net(self,X):
        X = X.view((-1, self.num_inputs))
        H = self.relu(torch.matmul(X, self.W1) + self.b1)
        return torch.matmul(H, self.W2) + self.b2

    # 模型评估
    def evaluate_accuracy(self,data_iter, net):
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n

    def output(self):
        # 定义模型参数
        self.W1 = torch.tensor(np.random.normal(0, 0.01, (self.num_inputs, self.num_hiddens)), dtype=torch.float)
        self.b1 = torch.zeros(self.num_hiddens, dtype=torch.float)
        self.W2 = torch.tensor(np.random.normal(0, 0.01, (self.num_hiddens, self.num_outputs)), dtype=torch.float)
        self.b2 = torch.zeros(self.num_outputs, dtype=torch.float)

        params = [self.W1, self.b1, self.W2, self.b2]
        for param in params:
            param.requires_grad_(requires_grad=True)

        # 定义损失函数
        LOSS = torch.nn.CrossEntropyLoss()

        # 模型训练
        tool.train_model(self.net, self.evaluate_accuracy, self.train_iter, self.test_iter, LOSS,
                         self.num_epochs, self.batch_size, params, self.lr)

        # 结果预测
        tool.predict_model(self.test_iter, self.net)
###########################################################################################################
#简洁实现
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
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
        model = nn.Sequential(
            FlattenLayer(),
            nn.Linear(self.num_inputs, self.num_hiddens),
            nn.ReLU(),
            nn.Linear(self.num_hiddens, self.num_outputs),
        )

        for params in model.parameters():
            init.normal_(params, mean=0, std=0.01)

        # softmax和交叉熵损失函数
        LOSS = nn.CrossEntropyLoss()
        # 定义优化算法
        OPTIMIZER = torch.optim.SGD(model.parameters(), lr=self.lr)
        # 模型训练
        tool.train_model(model, self.evaluate_accuracy, self.train_iter, self.test_iter, LOSS,
                         self.num_epochs, self.batch_size, None, None, OPTIMIZER)

        # 结果预测
        tool.predict_model(self.test_iter, model)
if __name__=='__main__':
    #ZERO().output()  # 输出从零实现的多层感知机模型训练结果
    CONCISE().output()  # 输出简洁实现的多层感知机模型训练结果
