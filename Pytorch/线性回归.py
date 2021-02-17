import torch
import random
###########################################################################################################
from matplotlib import pyplot as plt
from IPython import display
class INIT(object):
    def __init__(self):
        #初始化定义参数
        self.num_inputs = 2 #输入个数（特征数）
        self.num_examples = 1000 #样本数量
        self.true_w = torch.tensor([2, -3.4]) #真实权重
        self.true_b = 4.2 #真实偏置
        self.lr = 0.03 #学习率
        self.num_epochs = 3 #训练次数
        self.batch_size = 10 #训练批次大小

    # 构造样本数据y = Xw + b + noise
    def synthetic_data(self,w, b, num_examples):
        X = torch.normal(0, 1, (num_examples, len(w)))
        y = torch.matmul(X, w) + b
        y += torch.normal(0, 0.01, y.shape)
        y = y.reshape((-1, 1))
        return X, y

    def set_figsize(self, figsize=(3.5, 2.5)):
        # 用矢量图显示
        display.set_matplotlib_formats('svg')
        # 设置图的尺寸
        plt.rcParams['figure.figsize'] = figsize

    def output(self, printout='yes'):
        # 绘制样本散点图
        self.features, self.labels = self.synthetic_data(self.true_w, self.true_b, self.num_examples)
        if printout == 'yes':
            self.set_figsize()
            plt.scatter(self.features[:, 1], self.labels, 1)
            plt.show()

###########################################################################################################
#从零实现
class ZERO(INIT):
    def __init__(self):
        super(ZERO,self).__init__()

    # 读取数据
    def data_iter(self,batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
            yield features[batch_indices], labels[batch_indices]

    # 定义线性回归模型
    def linreg(self,X, w, b):
        return torch.matmul(X, w) + b

    # 定义平方损失函数
    def squared_loss(self,y_hat, y):
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

    # 定义优化算法（小批量随机梯度下降）
    def sgd(self,params, lr, batch_size):
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()

    def output(self, printout='yes'):
        super(ZERO, self).output('no')
        net = self.linreg
        loss = self.squared_loss

        # 初始化权重和偏置
        w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)

        for epoch in range(self.num_epochs):
            for X, y in self.data_iter(self.batch_size, self.features, self.labels):
                l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
                # Compute gradient on `l` with respect to [`w`, `b`]
                l.sum().backward()
                self.sgd([w, b], self.lr, self.batch_size)  # Update parameters using their gradient
            with torch.no_grad():
                if printout == 'yes':
                    train_l = loss(net(self.features, w, b), self.labels)
                    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
                    if epoch == 2:  # 训练第三次后输出
                        print('真实权重：{} 预测权重：{}'.format(self.true_w, w.numpy()))
                        print('真实偏置：{} 预测偏置：{}'.format(self.true_b, b.numpy()))
###########################################################################################################
#简洁实现
from torch.utils import data
from torch import nn
class CONCISE(INIT):
    def __init__(self):
        super(CONCISE,self).__init__()

    #读取数据
    def load_array(self,data_arrays, batch_size, is_train=True):
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    def output(self, printout='yes'):
        super(CONCISE, self).output('no')
        data_iter = self.load_array((self.features, self.labels), self.batch_size)

        # 定义线性回归模型
        net = nn.Sequential(nn.Linear(2, 1))
        net[0].weight.data.normal_(0, 0.01)
        net[0].bias.data.fill_(0)

        # 定义平方损失函数
        loss = nn.MSELoss()

        # 定义优化算法（随机梯度下降）
        trainer = torch.optim.SGD(net.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
            for X, y in data_iter:
                l = loss(net(X), y)
                trainer.zero_grad()
                l.backward()
                trainer.step()
            l = loss(net(self.features), self.labels)
            if printout == 'yes':
                print(f'epoch {epoch + 1}, loss {l:f}')
                if epoch == 2:  # 训练第三次后输出
                    print('真实权重：{} 预测权重：{}'.format(self.true_w, net[0].weight.data))
                    print('真实偏置：{} 预测偏置：{}'.format(self.true_b, net[0].bias.data))

###########################################################################################################
if __name__=='__main__':
    #INIT().output()#输出样本绘制的散点图
    #ZERO().output()#输出从零实现的线性回归模型训练结果
    CONCISE().output()  # 输出简洁实现的线性回归模型训练结果
