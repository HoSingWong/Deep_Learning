import tensorflow as tf
import sys
sys.path.append("..")
from tools import tensorflow_tool as tool
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#屏蔽通知和警告信息
###########################################################################################################
class INIT(object):
    def __init__(self):
        (train_images, self.train_labels), (test_images, self.test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_n,train_l,train_w=train_images.shape
        test_n,test_l,test_w=test_images.shape
        self.train_images = tf.reshape(train_images,(train_n,train_l,train_w, 1))
        self.test_images = tf.reshape(test_images, (test_n,test_l,test_w, 1))
###########################################################################################################
# 简洁实现
class CONCISE(INIT):
    def __init__(self):
        super(CONCISE, self).__init__()

    def output(self):
        #定义激活函数
        ACTION='sigmoid'
        # 定义和初始化模型
        net = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation=ACTION, input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation=ACTION),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation=ACTION),
            tf.keras.layers.Dense(84, activation=ACTION),
            tf.keras.layers.Dense(10, activation=ACTION)
        ])
        # softmax和交叉熵损失函数
        LOSS = 'sparse_categorical_crossentropy'
        # 定义优化算法
        OPTIMIZER  = tf.keras.optimizers.SGD(learning_rate=0.9, momentum=0.0, nesterov=False)

        net.compile(optimizer=OPTIMIZER,
                    loss=LOSS,
                    metrics=['accuracy'])
        net.fit(self.train_images, self.train_labels, epochs=5, validation_split=0.1)

        tool.predict_model((self.test_images, self.test_labels), net,TYPE='None')
if __name__=='__main__':
    CONCISE().output()  # 输出简洁实现的LeNet模型训练结果
