import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from utils import mnist_reader
import pickle
import time
import os


class LrScheduler:
    def __init__(self, init_lr=0.1, decay_rate=0.9, strategy="step", **args):
        self.init_lr = init_lr
        self.lr = init_lr
        conf_dict = {"exp": self.explr, "step": self.steplr}
        self.strategy_str = strategy
        self.strategy = conf_dict.get(strategy, self.explr)
        self.decay_rate = decay_rate
        self.conf = args
        self.epoch = 0

    @property
    def __str__(self):
        return f"LrScheduler(init_lr={self.init_lr}, decay_rate={self.decay_rate}, strategy=\"{self.strategy_str}\",**args={self.conf}"

    def update(self, epoch,batch):
        self.strategy(epoch,batch, **self.conf)
        return self.lr

    def explr(self, epoch,batch,**args):
        return self.init_lr * np.exp(-self.decay_rate * batch)

    def steplr(self, epoch,batch,**args):
        if epoch > self.epoch:
            # epoch增大时更新
            self.lr *= self.decay_rate
            self.epoch = epoch
        elif epoch < self.epoch:
            # 重置
            self.lr = self.init_lr
            self.epoch = epoch
        return self.lr


class SGD:
    def __init__(self, l2_reg=1e-4, lrscheduler=LrScheduler):
        self.lrscheduler = lrscheduler
        self.l2_reg = l2_reg
        self.decay_rate = 1 - l2_reg

    @property
    def __str__(self):
        return f"SGD(l2_reg={self.l2_reg}, lrscheduler={self.lrscheduler.__class__.__name__})"

    def update(self, epoch,batch, parameters):
        lr = self.lrscheduler.update(epoch,batch)
        # 权值衰减等价l2正则化
        for layer_paras in parameters:
            for para in layer_paras:  # W,b
                para.val *= self.decay_rate
                para.val -= lr * para.grad


class Parameter:
    def __init__(self, val, l2_reg=False):
        self.val = val
        self.grad = None
        self.l2_reg = l2_reg


class Layer:
    def forward(self, x):
        pass

    def backward(self):
        pass


class ActivationLayer(Layer):
    @property
    def __str__(self):
        return "%s()" % self.__class__.__name__


class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.W = Parameter(
            np.random.randn(input_size, output_size) * 2 / input_size**0.5)
        self.b = Parameter(np.zeros((1, self.output_size)))

    @property
    def __str__(self):
        return "Linear(*[%d, %d])" % (self.input_size, self.output_size)

    def forward(self, x):
        self.x = x
        self.A = np.matmul(x, self.W.val) + self.b.val
        return self.A

    def backward(self, grad):
        n = grad.shape[0]

        self.W.grad = np.matmul(self.x.T, grad) / n
        self.b.grad = np.sum(grad, axis=0, keepdims=True) / n
        return np.matmul(grad, self.W.val.T)


class Sigmoid(ActivationLayer):
    def forward(self, x):
        self.x = x
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, grad):
        # n*output_size
        return self.y * (1 - self.y) * grad


class Relu(ActivationLayer):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        grad[self.x <= 0] = 0
        return grad


class Softmax(ActivationLayer):
    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y = e_x / e_x.sum(axis=1, keepdims=True)
        return self.y

    def backward(self, grad):
        # large scale
        return np.matmul(np.diagflat(self.x) - np.dot(self.x, self.x.T), grad)


# 多分类任务，将不使用其他的loss
class CrossEntropyloss:
    def __init__(self):
        self.grad = None

    def compute_loss_acc(self, y_pred, y_true):
        # 输出层用softmax激活
        y_pred_soft = Softmax().forward(y_pred)
        # 直接得loss对softmax的梯度
        self.grad = y_pred_soft - y_true
        loss = -np.sum(y_true * np.log(y_pred_soft + 1e-15)) / y_true.shape[0]
        acc = (np.argmax(y_pred_soft, axis=1) == np.argmax(y_true,
                                                           axis=1)).mean()
        return loss, acc


class Net(Layer):
    def __init__(self, layers, optimizer, loss_cls=CrossEntropyloss):
        self.layers = layers
        self.loss_cls = loss_cls
        self.optimizer = optimizer
        self.best_model = deepcopy(self.layers)

        self.train_loss_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def print_layers(self):
        print("layers:")
        for layer in self.layers:
            print(layer)

    def get_settings(self):
        layers_str = "[%s]" % (", ".join(map(lambda x: x.__str__,
                                             self.layers)))

        result = f"""layers={layers_str},
epochs={self.epochs}, batch_size={self.batch_size},
loss_cls={self.loss_cls.__class__.__name__},
optimizer={self.optimizer.__str__},
lrscheduler={self.optimizer.lrscheduler.__str__}
"""
        return result

    @property
    def parameters(self):
        return [([layer.W, layer.b] if isinstance(layer, Linear) else [])
                for layer in self.layers]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self):
        # 其他loss类需定义好梯度的计算
        grad = self.loss_cls.grad
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        return grad

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        n = X_train.shape[0]

        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loss_list = []
        self.val_loss_list = []
        self.val_acc_list = []

        best_loss = float("+inf")
        best_model = deepcopy(self.layers)
        batch=0
        for epoch in range(epochs):
            print("Epoch %d/%d:" % (epoch + 1, epochs))
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                y_pred = self.forward(X_batch)
                loss, acc = self.loss_cls.compute_loss_acc(y_pred, y_batch)

                self.backward()
                self.optimizer.update(epoch,batch, self.parameters)
                if start > 0 and start % 500 == 0:
                    print("\tBatch %d/%d\tloss:%.4f\tacc: %.4f" %
                          (start, n, loss, acc))
                batch
            # 验证集
            y_val_pred = self.forward(X_val)
            val_loss, val_acc = self.loss_cls.compute_loss_acc(
                y_val_pred, y_val)
            print("\tVal_loss:%.4f  Val_acc: %.4f" % (val_loss, val_acc))
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.layers)

            self.train_loss_list.append(loss)
            self.val_loss_list.append(val_loss)
            self.val_acc_list.append(val_acc)
        self.best_model = best_model
        self.layers = deepcopy(best_model)
        return best_model

    def test(self, X_test, y_test):
        y_test_pred = self.forward(X_test)
        test_loss, test_acc = self.loss_cls.compute_loss_acc(
            y_test_pred, y_test)
        print("Test acc: %.4f" % test_acc)
        return test_acc

    def plot(self, figpath, showfig=False):
        epochs = len(self.train_loss_list)
        assert epochs > 0, "Error: Incomplete Trainning"
        x = np.arange(1, epochs + 1)
        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Train & Validation Loss")
        plt.plot(x, self.train_loss_list, color="black", label="Train")
        plt.plot(x, self.val_loss_list, color="blue", label="Validation")
        plt.legend()
        plt.savefig(figpath[:-7] + "_loss.png")

        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.plot(x, self.val_acc_list)
        plt.savefig(figpath[:-7] + "_val_acc.png")

        if showfig:
            plt.show()

    def weight_visualize(self, figpath):

        count = 0
        vmin, vmax = float("inf"), float("-inf")
        for layer in self.layers:
            if isinstance(layer, Linear):
                count += 1
                vmin = min(vmin, layer.W.val.min())
                vmax = max(vmax, layer.W.val.max())

        # cmap=plt.cm.binary
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        fig, ax = plt.subplots(count, 1, figsize=(18, 9))
        i = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                name = layer.__str__
                weights = layer.W.val
                im = ax[i].imshow(weights.T, cmap=cmap, norm=norm)
                ax[i].set_title(name)
                i += 1
                # print(weights.shape)
        fig.colorbar(im, ax=ax[-1], orientation="horizontal")
        plt.tight_layout()
        # plt.show()
        # print(load_path[:-7] + "_W_visualize.png")
        plt.savefig(figpath[:-7] + "_W_visualize.png")


def load_data(split_rate=0.7):
    def standardized(X):
        # X = X / 255.0
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        std[std == 0] = 1
        return (X - mean) / std

    def to_one_hot(y):
        num_classes = y.max() + 1
        y_one_hot = np.zeros((y.shape[0], num_classes))
        y_one_hot[np.arange(y.shape[0]), y] = 1
        return y_one_hot

    X_train_raw, y_train_raw = mnist_reader.load_mnist('data/fashion',
                                                       kind='train')
    X_test_raw, y_test_raw = mnist_reader.load_mnist('data/fashion',
                                                     kind='t10k')

    # 预处理
    indices = np.arange(X_train_raw.shape[0])
    np.random.shuffle(indices)
    X_train_raw = X_train_raw[indices]
    y_train_raw = y_train_raw[indices]

    X_train_raw = standardized(X_train_raw)
    X_test = standardized(X_test_raw)
    y_train_raw = to_one_hot(y_train_raw)
    y_test = to_one_hot(y_test_raw)

    # 划分训练集与验证集
    size = X_train_raw.shape[0]
    split_point = int(size * split_rate)
    # np.random.shuffle(X_train_raw)

    # X_train, y_train, X_val, y_val, X_test, y_test
    return [
        X_train_raw[:split_point], y_train_raw[:split_point],
        X_train_raw[split_point:], y_train_raw[split_point:], X_test, y_test
    ]


def create_net_para(
    layers,
    l2_reg=0.00001,
    init_lr=0.1,
    lr_decay_rate=0.9,
    lr_strategy="step",
):
    '''
        List[Layer] layers        : 神经网络的层次架构，注意相邻两层的形状关联，注意输出层默认连接了softmax层
        float       l2_reg        : l2正则化超参数
        float       init_lr       : 初始学习率
        float       lr_decay_rate : 学习率衰减因子
        str         lr_strategy   : 学习率衰减策略（目前支持"step"阶梯衰减、"exp"指数衰减）
    '''
    loss_cls = CrossEntropyloss()
    lr_sche = LrScheduler(init_lr=init_lr,
                          decay_rate=lr_decay_rate,
                          strategy=lr_strategy)
    optimizer = SGD(l2_reg=l2_reg, lrscheduler=lr_sche)
    net = Net(layers, optimizer, loss_cls)
    return net


def workflow(load_model=False,
             load_path=None,
             train_net=True,
             save_model=False,
             save_path=None,
             plot=False,
             showfig=False,
             weight_visualize=False,
             epochs=20,
             batch_size=64,
             **netargs):
    ''' 
        bool   load_model: 是否加载模型，False时将从netargs提取参数新建一个神经网络模型
        str    load_path : 加载模型的路径，例"./model/0428221405/model.pickle"
        bool   train_net : 是否训练模型，已加载的模型亦应支持继续训练（但不建议这样做）
        bool   save_model: 是否保存模型，保存的将会是整个Net类而非其参数，否则会丢失一些信息
        str    save_path : 保存模型的路径，不提供将自动生成在"./model/MMDDHHmmSS/model.pickle"
        bool   plot      : 是否绘制训练与验证的loss曲线图与验证的acc曲线图
        bool   showfig   : 绘制loss和acc图后是否展示（注意弹窗会阻塞进程）
        bool   weight_visualize: 是否作权重参数可视化，True时会保存可视化图至模型所在的路径下
        int    epochs    : 训练的批次数
        int    batch_size: 训练的batch大小
        
        netargs参见create_net_para，为该函数的全部参数
        
    '''
    np.random.seed(1024)
    timecheck = time.strftime("%m%d%H%M%S", time.localtime())
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    net = None
    if load_model:
        assert bool(load_path)
        with open(load_path, "rb") as f:
            net = pickle.load(f)
    else:
        net = create_net_para(**netargs)

    if train_net:
        net.train(X_train,
                  y_train,
                  X_val,
                  y_val,
                  epochs=epochs,
                  batch_size=batch_size)
    net.test(X_test, y_test)
    if not save_path:
        if load_path:
            save_path = load_path
        else:
            if not os.path.exists(f"./model"):
                os.makedirs(f"./model")
            os.chdir(f"./model")

            if not os.path.exists(f"./{timecheck}"):
                os.makedirs(f"./{timecheck}")
            os.chdir("..")
            save_path = f"./model/{timecheck}/model.pickle"
    if not save_path.endswith(".pickle"):
        save_path += ".pickle"
    if plot:
        net.plot(figpath=save_path, showfig=showfig)
    if weight_visualize:
        net.weight_visualize(save_path)

    if not train_net:
        return net
    if save_model:
        with open(save_path, "wb") as f:
            pickle.dump(net, f)
        with open(save_path[:-7] + "_settings.txt", "w",
                  encoding="utf-8") as f:
            f.write(net.get_settings())
    return net


if __name__ == "__main__":
    # (1)训练模型示例：
    # workflow(layers=[Linear(*[784, 512]),
    #                  Relu(),
    #                  Linear(*[512,256]),
    #                  Relu(),
    #                  Linear(*[256, 10])],
    #          epochs=50,
    #          batch_size=64,
    #          l2_reg=0.00001,
    #          init_lr=0.1,
    #          lr_decay_rate=0.9,
    #          lr_strategy="step",
    #          plot=True,
    #          weight_visualize=True,
    #          save_model=True)

    # (2)测试模型示例：
    # 由于train时会将loss和acc保存下来，故load模型可plot
    # workflow(load_model=True,
    #          load_path="./model/0512221405/model.pickle",
    #          train_net=False,
    #          plot=True,
    #          showfig=True)

    # (3)参数查找：
    # 在训练模型部分即可指定各超参数

    # (4)权重参数可视化：
    # workflow(load_model=True,
    #          load_path="./model/0512221405/model.pickle",
    #          train_net=False,
    #          weight_visualize=True)

    for timecheck in os.listdir("./model/"):
        print(timecheck)

        net=workflow(load_model=True,load_path="./model/%s/model.pickle"%timecheck,
                 train_net=False,weight_visualize=True)
        print(net.get_settings())
        print()