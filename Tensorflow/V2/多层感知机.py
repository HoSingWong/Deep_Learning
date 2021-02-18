import tensorflow as tf
import numpy as np
from tools import tensorflow_tool as tool

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#屏蔽通知和警告信息
###########################################################################################################
class INIT(object):
    def __init__(self):
        self.DATA=tool.DataLoader()
        self.batch_size=256#训练批次大小
        self.train_iter = tf.data.Dataset.from_tensor_slices((self.DATA.train_images, self.DATA.train_labels)).batch(self.batch_size)#训练集迭代器
        self.test_iter = tf.data.Dataset.from_tensor_slices((self.DATA.test_images, self.DATA.test_labels)).batch(self.batch_size)#测试集迭代器

        self.input_length=28#输入图片的长
        self.input_width=28#输入图片的宽
        self.num_outputs=10#输出结果类别
        self.num_epochs=5#训练次数
        self.lr =  0.1#学习率

        self.num_hiddens=256#隐藏层神经元
###########################################################################################################
#从零实现
class ZERO(INIT):
    def __init__(self):
        super(ZERO,self).__init__()

    #定义激活函数
    def relu(self,x):
        return tf.math.maximum(x, 0)

    #定义模型
    def net(self,X):
        X = tf.reshape(X, shape=[-1, self.num_inputs])
        h = self.relu(tf.matmul(X, self.W1) + self.b1)
        return tf.math.softmax(tf.matmul(h, self.W2) + self.b2)

    #定义损失函数
    def loss(self,y_hat, y_true):
        return tf.losses.sparse_categorical_crossentropy(y_true, y_hat)

    #模型评估
    def evaluate_accuracy(self,data_iter, net):
        acc_sum, n = 0.0, 0
        for _, (X, y) in enumerate(data_iter):
            y = tf.cast(y, dtype=tf.int64)
            acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
            n += y.shape[0]
        return acc_sum / n

    def output(self):
        #定义模型参数
        self.num_inputs=self.input_length*self.input_width
        self.W1 = tf.Variable(tf.random.normal(shape=(self.num_inputs, self.num_hiddens), mean=0, stddev=0.01, dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros(self.num_hiddens, dtype=tf.float32))
        self.W2 = tf.Variable(tf.random.normal(shape=(self.num_hiddens, self.num_outputs), mean=0, stddev=0.01, dtype=tf.float32))
        self.b2 = tf.Variable(tf.random.normal([self.num_outputs], stddev=0.1))

        # 训练模型
        trainer = tf.keras.optimizers.SGD(self.lr)
        tool.train_model(self.net,self.evaluate_accuracy, self.train_iter, self.test_iter, self.loss, self.num_epochs,
                         self.batch_size, [self.W1, self.b1,self.W2,self.b2], self.lr, trainer)
        # 结果预测
        tool.predict_model(self.test_iter, self.net)

###########################################################################################################
#简洁实现
class CONCISE(INIT):
    def __init__(self):
        super(CONCISE,self).__init__()
    def output(self):
        #定义和初始化模型
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.input_length, self.input_width)),
            tf.keras.layers.Dense(self.num_hiddens, activation='relu', ),
            tf.keras.layers.Dense(self.num_outputs, activation='softmax')
        ])
        # softmax和交叉熵损失函数
        LOSS = 'sparse_categorical_crossentropy'
        # 定义优化算法
        OPTIMIZER = tf.keras.optimizers.SGD(self.lr)
        # 训练模型
        model.compile(optimizer=OPTIMIZER,
                      loss=LOSS,
                      metrics=['accuracy'])
        # model.fit(self.DATA.train_images, self.DATA.train_labels, epochs=5,
        #           batch_size=self.batch_size,
        #           validation_data=(self.DATA.test_images, self.DATA.test_labels),
        #           validation_freq=1)  # 每多少次epoch迭代使用测试集验证一次结果
        model.fit(self.DATA.train_images, self.DATA.train_labels, epochs=5,
                  batch_size=self.batch_size,
                  validation_split=0.1)
        # 结果预测
        tool.predict_model(self.test_iter, model)

if __name__=='__main__':
    #ZERO().output()#输出从零实现的多层感知机模型训练结果
    CONCISE().output()  # 输出简洁实现的多层感知机模型训练结果
