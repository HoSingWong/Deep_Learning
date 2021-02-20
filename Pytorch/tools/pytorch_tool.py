import torch
from torch.utils import data
import torchvision
import sys
import matplotlib.pyplot as plt

#读取数据
def load_data_fashion_mnist(batch_size, resize=None, root='./Datasets/FashionMNIST'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter
############################################################################################################################
#模型训练
def train_model(net,evaluate_accuracy, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                """小批量随机梯度下降。"""
                with torch.no_grad():
                    for param in params:
                        param -= lr * param.grad / batch_size
                        param.grad.zero_()
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
############################################################################################################################
def train_model2(net,evaluate_accuracy, train_iter, test_iter, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    #定义损失函数
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc))
############################################################################################################################
#模型预测
def predict_model(test_iter,net,TYPE='iter',device=None):
    if TYPE=='iter':
        X, y = next(iter(test_iter))
    else:
        X, y = test_iter

    def get_fashion_mnist_labels(labels):
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]

    def show_fashion_mnist(images, labels):
        # 这里的_表示我们忽略（不使用）的变量
        _, figs = plt.subplots(1, len(images), figsize=(12, 12))
        for f, img, lbl in zip(figs, images, labels):
            f.imshow(img.view((images[0].shape[1], images[0].shape[2])).numpy())
            f.set_title(lbl)
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)
        plt.show()
    try:
        true_y=y.numpy()
    except:
        true_y =y
    try:
        pre_y=net(X).argmax(dim=1).numpy()
    except:
        try:
            pre_y=net(X).argmax(dim=1)
        except:
            pre_y = net(X.to(device)).argmax(dim=1)

    true_labels = get_fashion_mnist_labels(true_y)
    pred_labels = get_fashion_mnist_labels(pre_y)
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    show_fashion_mnist(X[0:9], titles[0:9])
