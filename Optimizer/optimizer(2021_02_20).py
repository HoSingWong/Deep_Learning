import matplotlib.pyplot as plt
import numpy as np
import math
#######################################################################################################################
#目标函数
#f(x) = x * x
def f(x):
    return x*x

#一维梯度下降
def gradient_descent(x,lr):
    return x - lr * (2 * x)  # f(x) = x * x的导数为f'(x) = 2 * x

#模拟训练，输入训练次数，优化器，学习率
def train(epoch, optimizer, lr):
    x = 10  # 初始化
    results = [x]
    for i in range(epoch):  # 迭代训练次数
        x = optimizer(x, lr)
        results.append(x)
        print('epoch {}, x:{}'.format(i, x))

    return results

#显示动态优化轨迹
def show_trace(f,result):
    n = max(abs(min(result)), abs(max(result)), 10)
    f_line = np.arange(-n, n, 0.1)

    plt.title('f(x) = x * x')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.ion()#打开界面
    plt.plot(f_line, [f(x) for x in f_line])
    X=[]
    Y=[]
    for i in range(len(result)):
        X.append(result[i])
        Y.append(f(result[i]))
        plt.plot(X, Y,'o-',color='orange' )
        plt.pause(0.5)#画图停顿时间
        plt.ioff()#关闭界面
    plt.show()

#######################################################################################################################
#目标函数
#f(x1,x2)=x1 ** 2 + 2 * x2 ** 2
def f_2d(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2

#多维梯度下降
def gradient_descent_2d(x1, x2, s1, s2,lr):
    return (x1 - lr * 2 * x1, x2 - lr * 4 * x2,0,0)

#随机梯度下降
def sgd_2d(x1, x2, s1, s2,lr):
    return (x1 - lr * (2 * x1 + np.random.normal(0.1)),
            x2 - lr * (4 * x2 + np.random.normal(0.1)),0,0)

#动量法
def momentum_2d(x1, x2,v1,v2,lr,gamma):
    v1 = gamma * v1 + lr * 2 * x1
    v2 = gamma * v2 + lr * 4 * x2
    return x1 - v1, x2 - v2,v1,v2

#AdaGrad
def adagrad_2d(x1, x2, s1, s2,lr):
    g1, g2, eps = 2 * x1, 4 * x2, 1e-6  # 前两项为自变量梯度
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= lr / math.sqrt(s1 + eps) * g1
    x2 -= lr / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

#RMSProp
def rmsprop_2d(x1, x2, s1, s2,lr,gamma):
    g1, g2, eps = 2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= lr / math.sqrt(s1 + eps) * g1
    x2 -= lr / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

#Adadelta
def adadelta_2d(x1, x2, s1, s2,delta1,delta2,gamma):
    g1, g2, eps = 2 * x1, 4 * x2, 1e-3#1e-5
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    g_1=math.sqrt(delta1+eps)/ math.sqrt(s1 + eps) * g1
    g_2 = math.sqrt(delta2 + eps) / math.sqrt(s2 + eps) * g2
    x1-=g_1
    x2-=g_2
    delta1=gamma * delta1+(1 - gamma)*g_1** 2
    delta2=gamma * delta2+(1 - gamma)*g_2** 2
    return x1, x2, s1, s2,delta1,delta2

#Adam
def adam_2d(x1, x2, s1, s2,v1,v2,t,lr,beta1=0.9,beta2=0.999):
    g1, g2, eps= 0.2 * x1, 4 * x2,1#1e-8
    v1 = beta1 * v1 + (1 - beta1) * g1
    v2 = beta1 * v2 + (1 - beta1) * g2
    s1 = beta2 * s1 + (1 - beta2) * g1 ** 2
    s2 = beta2 * s2 + (1 - beta2) * g2 ** 2
    v1=v1/(1 - beta1**t)
    v2=v2/(1 - beta1**t)
    s1=s1/(1 - beta2**t)
    s2=s2/(1 - beta2**t)
    g_1 = lr * v1 / (math.sqrt(s1) + eps)
    g_2 = lr * v2 / (math.sqrt(s2) + eps)
    x1-=g_1
    x2-=g_2
    t+=1
    return x1, x2, s1, s2,v1,v2,t

def adam(x1,x2,states,hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    p_params=[x1,x2]
    g_params=[2 * x1, 4 * x2]
    result=[]
    for p,g,(v, s) in zip(p_params,g_params, states):
        v = beta1 * v + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g ** 2
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p -= hyperparams['lr'] * v_bias_corr / (math.sqrt(s_bias_corr) + eps)
        result.append(p)
    hyperparams['t'] += 1
    return result[0],result[1]#x1,x2
#######################################################################################################################

def train_2d(epoch,optimizer,lr,gamma=None,adadelta=False,adam=False):
    x1, x2,s1,s2, v1, v2,delta1,delta2 = -5, -2,0,0,0,0,0,0
    results = [(x1, x2)]
    t=1
    for i in range(epoch):
        if gamma is None:
            if adam:
                x1, x2, s1, s2, v1, v2,t=optimizer(x1, x2,s1,s2, v1, v2,t,lr)
                #x1, x2 = adam(x1, x2, [[v1, s1], [v2, s2]], {'lr': lr, 't': t})
            else:
                x1, x2,s1,s2 = optimizer(x1, x2,s1,s2,lr)
        else:
            if adadelta:
                x1, x2, s1, s2,delta1,delta2 = optimizer(x1, x2, s1, s2,delta1,delta2, gamma)
            else:
                x1, x2,s1,s2 = optimizer(x1, x2,s1,s2, lr,gamma)
        results.append((x1, x2))
        print('epoch {}, x1:{}, x2:{}' .format(i, x1, x2))
    return results

def show_trace_2d(f, results):
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2),colors='#1f77b4')

    plt.title('f(x1,x2)=x1 ** 2 + 2 * x2 ** 2')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.ion()  # 打开界面
    X = []
    Y = []
    for x, y in results:
        X.append(x)
        Y.append(y)
        plt.plot(X, Y, 'o-', color='orange')
        plt.pause(0.5)  # 画图停顿时间
        plt.ioff()  # 关闭界面
    plt.show()

#######################################################################################################################
if __name__=='__main__':
    lr=0.2
    epoch=10
    gamma=0.8
    #show_trace(f,train(epoch,gradient_descent,lr))#一维梯度下降
    show_trace_2d(f_2d,train_2d(epoch,gradient_descent_2d,lr))#多维梯度下降
    #show_trace_2d(f_2d, train_2d(epoch,sgd_2d,lr))#随机梯度下降
    #show_trace_2d(f_2d, train_2d(epoch,momentum_2d,lr,gamma=gamma))#动量法
    #show_trace_2d(f_2d, train_2d(epoch,adagrad_2d,lr))#AdaGrad
    #show_trace_2d(f_2d, train_2d(epoch,rmsprop_2d,lr,gamma=gamma))#RMSProp
    #show_trace_2d(f_2d, train_2d(epoch, adadelta_2d,lr, gamma=gamma,adadelta=True))#AdaDelta
    #show_trace_2d(f_2d, train_2d(epoch, adam_2d, lr, adam=True))#Adam

    # x1, x2, s1, s2, v1, v2= -5, -2, 0, 0, 0, 0
    # t=1
    # results = [(x1, x2)]
    # for i in range(1000):
    #     x1,x2=adam(x1, x2, [[v1, s1], [v2, s2]], {'lr': lr, 't': t})
    #     results.append((x1, x2))
    # print('epoch {}, x1:{}, x2:{}'.format(i, x1, x2))
