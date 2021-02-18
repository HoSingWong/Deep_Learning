import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

class DataLoader(object):
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        self.train_images = np.expand_dims(train_images.astype(np.float32)/255.0,axis=-1)#(60000, 28, 28, 1)
        self.test_images = np.expand_dims(test_images.astype(np.float32)/255.0,axis=-1)#(10000, 28, 28, 1)
        self.train_labels = train_labels.astype(np.int32)
        self.test_labels = test_labels.astype(np.int32)
        self.num_train, self.num_test = self.train_images.shape[0], self.test_images.shape[0]

    def get_batch_train(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        #need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.train_images[index],224,224,)
        return resized_images.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        #need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.test_images[index],224,224,)
        return resized_images.numpy(), self.test_labels[index]

#################################################################################################
# 模型训练
def train_model(net,evaluate_accuracy, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = tf.reduce_sum(loss(y_hat, y))
            grads = tape.gradient(l, params)
            if trainer is None:
                # 如果没有传入优化器，则使用原先编写的小批量随机梯度下降
                for i, param in enumerate(params):
                    param.assign_sub(lr * grads[i] / batch_size)
            else:
                # tf.keras.optimizers.SGD 直接使用是随机梯度下降 theta(t+1) = theta(t) - learning_rate * gradient
                # 这里使用批量梯度下降，需要对梯度除以 batch_size, 对应原书代码的 trainer.step(batch_size)
                trainer.apply_gradients(zip([grad / batch_size for grad in grads], params))

            y = tf.cast(y, dtype=tf.float32)
            train_l_sum += l.numpy()
            train_acc_sum += tf.reduce_sum(
                tf.cast(tf.argmax(y_hat, axis=1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
        epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
#################################################################################################
#模型预测
def predict_model(test_iter,net,TYPE='iter'):
    if TYPE=='iter':
        X, y = next(iter(test_iter))
    else:
        X, y = test_iter

    def get_fashion_mnist_labels(labels):
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]

    def show_fashion_mnist(images, labels):
        # 这⾥的_表示我们忽略（不使⽤）的变量
        _, figs = plt.subplots(1, len(images), figsize=(12, 12)) # 这里注意subplot 和subplots 的区别
        for f, img, lbl in zip(figs, images, labels):
            f.imshow(tf.reshape(img, shape=(28, 28)).numpy())
            f.set_title(lbl)
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)
        plt.show()
    try:
        true_y=y.numpy()
    except:
        true_y =y
    try:
        pre_y=tf.argmax(net(X), axis=1).numpy()
    except:
        pre_y=tf.argmax(net(X), axis=1)
    true_labels = get_fashion_mnist_labels(true_y)
    pred_labels = get_fashion_mnist_labels(pre_y)
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    show_fashion_mnist(X[0:9], titles[0:9])
