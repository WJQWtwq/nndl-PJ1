# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.

import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment，确保多次试验同一条件
np.random.seed(309)

train_images_path = r'.\codes\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\codes\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)     #随机处理后的1-60000序号'腌渍'起来(pickle)持久保存，数据序列化
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:5000] #验证集图像(随机后的图片序列前10000个)
valid_labs = train_labs[:5000] #验证集标签
train_imgs = train_imgs[5000:] #训练集图像
train_labs = train_labs[5000:] #训练集标签

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# 定义前馈神经网络模型
# 输入层大小train_imgs.shape[-1](train_imgs的最后一维大小)，隐藏层(单层)600个神经元，输出层10个神经元(十分类)
# ReLU激活函数，lambda_list权重衰减系数（两个，一个对隐藏层，一个对输出层）

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 600, 10], 'ReLU', [1e-4, 1e-4,1e-4,1e-4])
#CNN_model = nn.models.Model_CNN([28,600,10],  ['None','ReLU','None'], [1e-4, 1e-4], kernel_size=20, kernel_num=3)

# 优化器SGD，初始学习率0.06
optimizer = nn.optimizer.MomentGD(init_lr=0.05, model=linear_model)

# 学习率调度器MultiStepLR
# 训练的第 800、2400 和 4000 次迭代时调整学习率，每次减半
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[600, 900, 1200], gamma=0.5)
#scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[200, 400, 1000], gamma=0.5)
#scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer)

# 多分类交叉熵损失函数
#loss_fn = nn.op.MultiCrossEntropyLoss(model=CNN_model, max_classes=train_labs.max()+1)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

# 训练器，管理模型的训练过程
# 用准确率nn.metric.accuracy作评价指标，并绑定前面的模型四要素，batch_size未传递默认2^5=32
runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler,batch_size=32, model_type='linear')

# 开始训练
# num_epochs训练轮数，log_iters打印日志间隔，save_dir保存目录
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=1, log_iters=5, save_dir=r'.\codes\best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()

