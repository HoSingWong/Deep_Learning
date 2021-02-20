import numpy as np
def linear(x):
    return x


def sigmoid(x):
    ex = np.exp(x)
    return ex / (ex + 1)


def tanh(x):
    e2x = np.exp(2 * x)
    return (e2x - 1) / (e2x + 1)

    #return np.tanh(x)


def softmax(x):
    D = np.max(x)
    exp_x = np.exp(x-D)
    
    return exp_x / np.sum(exp_x)


def softplus(x):
    return np.log(np.exp(x) + 1)


def softsign(x):
    return x / (np.abs(x) + 1)


def elu(x):
    return np.where(x < 0, np.exp(x) - 1, x)


def relu(x):
    return np.where(x > 0, x, 0)


def relu6(x):
    maxx = np.where(x > 0, x, 0)
    return np.where(maxx < 6, maxx, 6)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    return np.where(x < 0, scale * alpha * (np.exp(x) - 1), scale * x)


def leaky_relu(x, alpha=0.2):
    return np.where(x < 0, alpha * x, x)


def swish(x, beta=1.0):
    ex = np.exp(beta * x)

    return (ex / (ex + 1)) * x


def mish(x):
    return x * tanh(np.log(1+np.exp(x)))

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    colors = ['k', 'm', 'b', 'g', 'c', 'r',
              '-.m', '-.b', '-.g', '-.c', '-.r',
              '-.k','--k']
    activations = ['linear', 'tanh', 'sigmoid', 'softplus', 'softsign',
                   'elu', 'relu', 'selu', 'relu6', 'leaky_relu', 'swish',
                   'softmax','mish']
    
    x = np.linspace(-10, 10, 200)

    # for activation in activations:
    #     print("---show activation: " + activation + "---")
    #     y = globals()[activation](x)
    #     plt.figure()
    #     plt.plot(x, y, 'r')
    #     plt.title(activation + ' activation')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.grid()
    #     plt.show()

    plt.figure()
    #第一种：全部输出
    # for activation, color in zip(activations, colors):
    #     print("---show activation: " + activation + "---")
    #     y = globals()[activation](x)
    #     plt.plot(x, y, color)
    # plt.legend(activations)
    # plt.title('activation')
    #第二种：特定输出
    color='red'
    activation='mish'
    print("---show activation: " + activation + "---")
    y = globals()[activation](x)
    plt.plot(x, y, color)
    plt.title(activation)



    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()
