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

###########################################################################################################
#从零实现
class ZERO(INIT):
    def __init__(self):
        super(ZERO,self).__init__()

    #实现softmax运算
    def softmax(self,logits, axis=-1):
        return tf.exp(logits)/tf.reduce_sum(tf.exp(logits), axis, keepdims=True)

    #定义模型
    def net(self,X):
        logits = tf.matmul(tf.reshape(X, shape=(-1, self.W.shape[0])), self.W) + self.b
        return self.softmax(logits)


    #定义损失函数
    def cross_entropy(self,y_hat, y):
        y = tf.cast(tf.reshape(y, shape=[-1, 1]),dtype=tf.int32)
        y = tf.one_hot(y, depth=y_hat.shape[-1])
        y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]),dtype=tf.int32)
        return -tf.math.log(tf.boolean_mask(y_hat, y)+1e-8)

    #计算分类准确率
    def accuracy(self,y_hat, y):
        return np.mean((tf.argmax(y_hat, axis=1) == y))

    #模型评估
    def evaluate_accuracy(self,data_iter, net):
        acc_sum, n = 0.0, 0
        for _, (X, y) in enumerate(data_iter):
            y = tf.cast(y,dtype=tf.int64)
            acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
            n += y.shape[0]
        return acc_sum / n

    def output(self):
        #初始化模型参数
        num_inputs =self.input_length*self.input_width
        self.W = tf.Variable(tf.random.normal(shape=(num_inputs, self.num_outputs), mean=0, stddev=0.01, dtype=tf.float32))
        self.b = tf.Variable(tf.zeros(self.num_outputs, dtype=tf.float32))
        #训练模型
        trainer = tf.keras.optimizers.SGD(self.lr)
        tool.train_model(self.net,self.evaluate_accuracy, self.train_iter, self.test_iter, self.cross_entropy, self.num_epochs, self.batch_size, [self.W, self.b], self.lr,trainer)
        #结果预测
        tool.predict_model(self.test_iter,self.net)

###########################################################################################################
#简洁实现
class CONCISE(INIT):
    def __init__(self):
        super(CONCISE,self).__init__()
    def output(self):
        #定义和初始化模型
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.input_length, self.input_width)),
            tf.keras.layers.Dense(self.num_outputs, activation=tf.nn.softmax)
        ])
        #softmax和交叉熵损失函数
        LOSS = 'sparse_categorical_crossentropy'
        #定义优化算法
        OPTIMIZER = tf.keras.optimizers.SGD(self.lr)
        #训练模型
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

###########################################################################################################
if __name__=='__main__':
    #ZERO().output()#输出从零实现的Softmax回归模型训练结果
    CONCISE().output()  # 输出简洁实现的Softmax回归模型训练结果
