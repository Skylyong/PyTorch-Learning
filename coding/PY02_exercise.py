import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 练习，利用pytorch实现简单的全连接神经网络
'''
该神经网络包含一个输入层，一个中间隐藏层和一个输出层，隐藏层的激活函数选用relu，输出层的激活函数为线性激活函数
输入层维度：input_dim
隐藏层维度：hidden_dim
输出层维度:out_dim
'''


class CreateDataSet(object):
    def oracle(self, x, landmarks):
        dists = cdist(x, landmarks, metric='euclidean')
        d1 = dists.max(axis=1)
        d2 = dists.min(axis=1)
        return d1 * d2

    def oracle_classification(self, X, pos_landmarks, neg_landmarks):
        pos_value = self.oracle(X, pos_landmarks)
        neg_value = self.oracle(X, neg_landmarks)
        y_ = (pos_value <= neg_value).astype(int)
        y = np.zeros((y_.shape[0], 2))
        y[np.arange(y.shape[0]), y_] = 1
        return y

    def make_dataset_classification(self, size=100, complexity=2, ndim=3, return_landmarks=False):
        data_mtx = np.random.rand(size, ndim)
        pos_landmarks = np.random.rand(complexity, ndim)
        neg_landmarks = np.random.rand(complexity, ndim)
        y = self.oracle_classification(data_mtx, pos_landmarks, neg_landmarks)
        if return_landmarks:
            return data_mtx, y, pos_landmarks, neg_landmarks
        else:
            return data_mtx, y

    def make_2d_grid_dataset_classification(self, size, pos_landmarks, neg_landmarks):
        x = np.linspace(0.0, 1.0, int(size ** 0.5))
        y = np.linspace(0.0, 1.0, int(size ** 0.5))
        xx, yy = np.meshgrid(x, y)
        z = np.dstack((xx, yy))
        data_mtx = z.reshape(-1, 2)
        y = self.oracle_classification(data_mtx, pos_landmarks, neg_landmarks)
        return data_mtx, y

    def plot_2d_classification(self, X_test, y_test, preds, pos_landmarks, neg_landmarks):
        y_test = np.argmax(y_test, axis=1)
        preds = np.argmax(preds, axis=1)
        acc = np.sum(y_test == preds) / y_test.shape[0]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(18, 7)
        ax1.set_title("Prediction")
        ax2.set_title("Truth")
        ax3.set_title(f"Comparsion acc:{acc}")

        ax1.scatter(X_test[:, 0], X_test[:, 1], c=preds, marker='o', cmap=plt.cm.coolwarm, alpha=0.2)
        ax1.scatter(pos_landmarks[:, 0], pos_landmarks[:, 1], s=200, c='b', marker='*', alpha=0.65)
        ax1.scatter(neg_landmarks[:, 0], neg_landmarks[:, 1], s=200, c='r', marker='*', alpha=0.65)

        ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', cmap=plt.cm.coolwarm, alpha=0.2)
        ax2.scatter(pos_landmarks[:, 0], pos_landmarks[:, 1], s=200, c='b', marker='*', alpha=0.65)
        ax2.scatter(neg_landmarks[:, 0], neg_landmarks[:, 1], s=200, c='r', marker='*', alpha=0.65)

        ax3.scatter(X_test[:, 0], X_test[:, 1], c=preds, marker='o', cmap=plt.cm.coolwarm, alpha=0.2)
        ax3.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', cmap=plt.cm.coolwarm, alpha=0.2)
        ax3.scatter(pos_landmarks[:, 0], pos_landmarks[:, 1], s=200, c='b', marker='*', alpha=0.65)
        ax3.scatter(neg_landmarks[:, 0], neg_landmarks[:, 1], s=200, c='r', marker='*', alpha=0.65)
        plt.show()


class ExerciseNet(object):
    def __init__(self, input_dim, hidden_dim, out_dim, is_bias=True):
        '''
        :param input_dim: 输入向量维度
        :param hidden_dim: 中间层维度
        :param out_dim: 输出向量维度
        :param is_bias: 是否有bias
        '''
        self.model = self._init_model_paramers_(input_dim, hidden_dim, out_dim, is_bias)
        self.is_bias = is_bias
        self.official_loss = torch.nn.CrossEntropyLoss()

    def _init_model_paramers_(self, input_dim, hidden_dim, out_dim, is_bias):
        '''
        :param input_dim: 输入向量维度
        :param hidden_dim: 中间层维度
        :param out_dim: 输出向量维度
        :param is_bias: 是否有bias
        :return: 一个字典，其值为模型的参数矩阵
        '''
        model = dict()
        if is_bias:
            model['Wh'] = torch.rand(size=(hidden_dim, input_dim + 1), requires_grad=True)
            model['Wo'] = torch.rand(size=(out_dim, hidden_dim + 1), requires_grad=True)
        else:
            model['Wh'] = torch.rand(size=(hidden_dim, input_dim), requires_grad=True)
            model['Wo'] = torch.rand(size=(out_dim, hidden_dim), requires_grad=True)
        return model

    def _relu_(self, x):
        '''
        :param x: 输入矩阵，行表示样本个数，列表示维度
        :return: relu激活函数运算后的值
        '''
        out = x.clone()
        out[out < 0] = 0
        return out

    def _softmax_(self, x):
        '''
        :param x: 输入矩阵，行表示样本个数，列表示维度
        :return: softmax运算后的值
        '''
        x = x - torch.max(x)
        return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)

    def _crossentropy_loss(self, y, t):
        '''
        :param y: 样本的预测值，行表示样本个数，列表示样本在各个类别上的概率分布
        :param t: 样本真实标签，行表示样本个数，列是一个one-hot向量
        :return: 批样本的交叉熵损失
        '''
        y = self._softmax_(y)
        return -torch.sum(t * torch.log(y + 1e-7))

    def _update_weights_(self, lr):
        '''
        :param lr: 学习率
        :return: None
        '''
        for key in self.model.keys():
            self.model[key].data -= lr * self.model[key].grad

    def _zero_grad_(self):
        '''
        将梯度置为0
        '''
        for key in self.model.keys():
            self.model[key].grad = None

    def forward(self, x):
        '''
        前向传播计算
        :param x: T行input_dim列，T表示样本的个数，input_dim表示样本的维度
        :return: 全连接网络的输出值，有T行out_dim列
        '''
        if self.is_bias:
            x = torch.cat((x, torch.ones(x.size()[0], 1)), dim=1)
            h_values = torch.mm(self.model['Wh'], x.t())
            h_values = self._relu_(h_values)
            h_values = torch.cat((h_values, torch.ones(1, h_values.size()[1])), dim=0)  # h_values增一维
            out = torch.mm(self.model['Wo'], h_values)
        else:
            h_values = torch.mm(self.model['Wh'], x.t())
            h_values = self._relu_(h_values)
            out = torch.mm(self.model['Wo'], h_values)
        return out.t()

    def _train_(self, x_train, y_train, batch_size=16, lr=0.01):
        '''
        训练一轮网络
        :param x_train: 输入特征, 行表示样本个数，列表示样本维度
        :param y_train: 真实标签，行表示样本个数，列为one-hot向量
        :param batch_size: 批大小
        :param lr: 学习率
        :return: 当前批训练的平均loss
        '''
        index_arr = torch.randperm(x_train.size()[0])
        total_loss = []
        for step in range(0, x_train.size()[0], batch_size):
            begin, end = step, min(step + batch_size, x_train.size()[0])
            batch_index = index_arr[begin:end]
            y_pred = self.forward(x_train[batch_index])
            loss = self.loss(y_pred, y_train[batch_index])
            self._zero_grad_()
            loss.backward()
            self._update_weights_(lr=lr)
            total_loss.append(loss.item())
        return np.mean(np.array(total_loss))

    def train(self, x_train, y_train, batch_size=16, lr=0.01, epochs=20, loss_type='custom'):
        '''
        训练多轮网络
        :param x_train: 输入特征, 行表示样本个数，列表示样本维度
        :param y_train: 真实标签，行表示样本个数，列为one-hot向量
        :param batch_size: 批大小
        :param lr: 学习率
        :param epochs: 学习轮次
        :param loss_type: 使用何种交叉熵loss，值为custom使用自定义的loss，否则使用官方库定义的loss
        :return: None
        '''
        if loss_type == 'custom':
            self.loss = self._crossentropy_loss
            print(f'use loss: {self.loss.__name__}')
        else:
            self.loss = self.official_loss
            print(f'use loss: {type(self.loss).__name__}')
        x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        for epoch in range(epochs):
            loss = self._train_(x_train, y_train, batch_size=batch_size, lr=lr)
            print(f'epoch:{epoch}, loss:{loss}')

    def test(self, x_test, y_test):
        '''
        测试网络效果，评估指标为准确率
        :param x_test: 输入特征, 行表示样本个数，列表示样本维度
        :param y_test: 真实标签，行表示样本个数，列为one-hot向量
        :return: None
        '''
        with torch.no_grad():
            x_test, y_test = torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
            y_pred = self.forward(x_test)
            y_pred = torch.argmax(y_pred, dim=1)
            acc = torch.sum(y_pred == torch.argmax(y_test, dim=1)) / y_test.size()[0]
            print(f'acc: {acc}')

    def pred(self, x_test):
        '''
        用训练好的模型做预测
        :param x_test: 输入特征, 行表示样本个数，列表示样本维度
        :return: 预测矩阵，行表示样本个数，列为样本在类别上面的概率分布
        '''
        with torch.no_grad():
            x_test = torch.tensor(x_test, dtype=torch.float32)
            y_pred = self.forward(x_test)
            return y_pred.numpy()


# if __name__=="__main__":
data_set = CreateDataSet()
x_train, y_train, pos_landmarks, neg_landmarks = data_set.make_dataset_classification(size=300, complexity=2,
                                                                                      ndim=2, return_landmarks=True)
x_test, y_test = data_set.make_2d_grid_dataset_classification(3000, pos_landmarks, neg_landmarks)
net = ExerciseNet(2, 5, 2)
net.train(x_train, y_train, lr=0.05, epochs=10, loss_type='custom')
net.test(x_test, y_test)
preds = net.pred(x_test)
data_set.plot_2d_classification(x_test, y_test, preds, pos_landmarks, neg_landmarks)
