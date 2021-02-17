import tensorflow as tf
import random
###########################################################################################################
from matplotlib import pyplot as plt
from IPython import display
class INIT(object):
    def __init__(self):
        #初始化定义参数
        self.num_inputs = 2 #输入个数（特征数）
        self.num_examples = 1000 #样本数量
        self.true_w = tf.constant([2, -3.4]) #真实权重
        self.true_b = 4.2 #真实偏置
        self.lr = 0.03 #学习率
        self.num_epochs = 3 #训练次数
        self.batch_size = 10 #训练批次大小

    #构造样本数据y = Xw + b + noise
    def synthetic_data(self,w, b, num_examples):
        X = tf.zeros((num_examples, w.shape[0]))
        X += tf.random.normal(shape=X.shape)
        y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
        y += tf.random.normal(shape=y.shape, stddev=0.01)
        y = tf.reshape(y, (-1, 1))
        return X, y

    def set_figsize(self,figsize=(3.5, 2.5)):
        # 用矢量图显示
        display.set_matplotlib_formats('svg')
        # 设置图的尺寸
        plt.rcParams['figure.figsize'] = figsize

    def output(self,printout='yes'):
        #绘制样本散点图
        self.features, self.labels = self.synthetic_data(self.true_w, self.true_b, self.num_examples)
        if printout=='yes':
            self.set_figsize()
            plt.scatter(self.features[:, 1], self.labels, 1)
            plt.show()

###########################################################################################################
#从零实现
class ZERO(INIT):
    def __init__(self):
        super(ZERO,self).__init__()

    #读取数据
    def data_iter(self,batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            j = tf.constant(indices[i: min(i + batch_size, num_examples)])
            yield tf.gather(features, j), tf.gather(labels, j)

    #定义线性回归模型
    def linreg(self,X, w, b):
        return tf.matmul(X, w) + b

    #定义平方损失函数
    def squared_loss(self,y_hat, y):
        return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2

    #定义优化算法（小批量随机梯度下降）
    def sgd(self,params, grads, lr, batch_size):
        for param, grad in zip(params, grads):
            param.assign_sub(lr*grad/batch_size)

    def output(self,printout='yes'):
        super(ZERO, self).output('no')
        net = self.linreg
        loss = self.squared_loss

        # 初始化权重和偏置
        w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01), trainable=True)
        b = tf.Variable(tf.zeros(1), trainable=True)

        for epoch in range(self.num_epochs):
            for X, y in self.data_iter(self.batch_size, self.features, self.labels):
                with tf.GradientTape() as g:
                    l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
                # Compute gradient on l with respect to [`w`, `b`]
                dw, db = g.gradient(l, [w, b])
                # Update parameters using their gradient
                self.sgd([w, b], [dw, db], self.lr, self.batch_size)
            train_l = loss(net(self.features, w, b), self.labels)
            if printout == 'yes':
                print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
                if epoch == 2:  # 训练第三次后输出
                    print('真实权重：{} 预测权重：{}'.format(self.true_w, w.numpy()))
                    print('真实偏置：{} 预测偏置：{}'.format(self.true_b, b.numpy()))

###########################################################################################################
#简洁实现
class CONCISE(INIT):
    def __init__(self):
        super(CONCISE,self).__init__()

    #读取数据
    def load_array(self,data_arrays, batch_size, is_train=True):
        dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
        if is_train:
            dataset = dataset.shuffle(buffer_size=self.num_examples)
        dataset = dataset.batch(batch_size)
        return dataset

    def output(self,printout='yes'):
        super(CONCISE, self).output('no')
        # `keras` is the high-level API for TensorFlow
        data_iter = self.load_array((self.features, self.labels), self.batch_size)

        # 定义线性回归模型
        initializer = tf.initializers.RandomNormal(stddev=0.01)
        net = tf.keras.Sequential()
        net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))

        # 定义平方损失函数
        loss = tf.keras.losses.MeanSquaredError()

        # 定义优化算法（小批量随机梯度下降）
        trainer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        for epoch in range(self.num_epochs):
            for X, y in data_iter:
                with tf.GradientTape() as tape:
                    l = loss(net(X, training=True), y)
                # Compute gradient on l with respect to [`w`, `b`]
                grads = tape.gradient(l, net.trainable_variables)
                # Update parameters using their gradient
                trainer.apply_gradients(zip(grads, net.trainable_variables))
            l = loss(net(self.features), self.labels)
            if printout == 'yes':
                print(f'epoch {epoch + 1}, loss {l:f}')
                if epoch==2:#训练第三次后输出
                    print('真实权重：{} 预测权重：{}'.format(self.true_w, net.get_weights()[0]))
                    print('真实偏置：{} 预测偏置：{}'.format(self.true_b, net.get_weights()[1]))

###########################################################################################################
if __name__=='__main__':
    #INIT().output()#输出样本绘制的散点图
    #ZERO().output()#输出从零实现的线性回归模型训练结果
    CONCISE().output()#输出简洁实现的线性回归模型训练结果
