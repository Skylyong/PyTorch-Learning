## PY02 pytorch基础知识

### 2.1 张量  

Tensor和numpy中的多维数组十分类似，与numpy中的多维数组不同的是，Tensor提供GPU计算和自动梯度计算，这使得以Tensor为基础单元构建深度学习模型不仅能够提升计算速度，还能够大大减少编程复杂度。

1. 常见的构造tensor的方法

  ```python
  # 1. 常见的构造tensor的方法
  Tensor(sizes)
  tensor(data)
  ones(sizes)
  zeros(sizes)
  eye(sizes)
  arange(s,e,step) # 从s到e，步长为step
  linspace(s,e,steps) #从s到e，均匀分成steps份
  rand/randn(sizes) #rand是[0,1)均匀分布；randn是服从N(0，1)的正态分布
  normal(mean,std) #正态分布(均值为mean，标准差是std)
  randperm(m) #随机排列
  ```

2. 张量操作

   张量的操作包括：四则运算，索引操作，维度变换，取值操作。
   
   其中索引方法与numpy类似，需要注意的是，索引出来的结果与原数据共享内存，可以使用copy方法开辟新的内存使得索引出来的结果与原数据“解绑”。
   
   pytorch中的维度变换使用view方法和reshape方法实现，view方法得到的新tensor与原数据共享内存。reshape方法在改变维度的同时开辟新内存存放数据，但是并不确保返回的是拷贝值，因此不推荐使用。如果需要改变维度的同时开辟新的内存，正确做法是：用clone方法构建张量副本，然后用view方法进行维度变换。
   
   我们使用item方法来获取tensor对应的值。
   
   ```python
   # 加法操作
   y = torch.rand(4, 3) 
   x = torch.rand(4, 3) 
   
   x+y # 方式1
   torch.add(x, y) #方式2
   y.add_(x) # 方式3，原值修改
   ```
   
3. 张量的广播机制

   与numpy中的多维数组一样，tensor具备广播机制。

### 2.2 自动求导 

​	当设置Tensor的requires_grad属性为True的时候，pytorch会追踪对该张量的所有操作，一次正向传播计算结束之后，可以用backward方法来计算梯度，张量的梯度会累加到grad属性中。

因为追踪张量的梯度会占用额外的内存资源，当我们不需要追踪张量的梯度时，可以用detach方法将张量从计算图中剥离。也可以用with torch.no_grad()对代码块进行包装。

Tensor还有一个特别重要的属性:grad_fn, 该属性指向创建Tensor的函数，如果Tensor不是由函数生成的，则其值为None，否则为函数对象的地址。

假设当前的张量为out，当我们调用out.backward()方法求解计算图中张量的梯度时。若out.item()是一个标量，backward()方法不需要传入任何参数；否则，需要传入一个与out同形的参数。推理过程如下：

```bash
# TODO 此处有若干关于计算图的知识待补充
```



### 2.3 并行计算 

1. 数据在gpu和cpu之间切换时很耗时，应该尽量避免；
2. gpu运算很快，但是简单的操作应该尽量使用cpu完成
3. 指定gpu命令

```bash
 # 方法一： 设置在文件最开始部分
import os
os.environ["CUDA_VISIBLE_DEVICE"] = "2" # 设置默认的显卡

# 方法二： 运行代码时指定
CUDA_VISBLE_DEVICE=0,1 python train.py # 使用0，1两块GPU
```

### 2.4 综合练习：利用pytorch实现全连接神经网络      

该神经网络包含一个输入层，一个中间隐藏层和一个输出层，隐藏层的激活函数选用relu，输出层的激活函数为线性激活函数，**除梯度计算采用自动梯度外，其他功能均需要自己实现**。
输入层维度：input_dim
隐藏层维度：hidden_dim
输出层维度: out_dim      

代码如下：

```python
'''
@作者： leon
@完成时间：2022年07月13日
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

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


# test
data_set = CreateDataSet()
x_train, y_train, pos_landmarks, neg_landmarks = data_set.make_dataset_classification(size=300, complexity=2,
                                                                                      ndim=2, return_landmarks=True)
x_test, y_test = data_set.make_2d_grid_dataset_classification(3000, pos_landmarks, neg_landmarks)
net = ExerciseNet(2, 5, 2)
net.train(x_train, y_train, lr=0.05, epochs=10, loss_type='custom')
net.test(x_test, y_test)
preds = net.pred(x_test)
data_set.plot_2d_classification(x_test, y_test, preds, pos_landmarks, neg_landmarks)
```



### 参考资料

- https://datawhalechina.github.io/thorough-pytorch/index.html

- https://pytorch.org/docs/stable/tensors.html

- https://zhuanlan.zhihu.com/p/393041305?utm_source=wechat_session&utm_medium=social&utm_oi=910884195482599424&utm_campaign=shareopn

  

