import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None, model_type='linear', reg=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.model_type = model_type
        self.reg = reg

        self.train_scores = []
        self.dev_scores = []   # 训练集和验证集估计指标
        self.train_loss = []
        self.dev_loss = []     # 训练集和验证集损失值

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0) #右边为默认值
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):  # 若保存目录不存在，则创建
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]  # 确保特征和标签数量一致

            idx = np.random.permutation(range(X.shape[0])) #生成随机索引

            X = X[idx]
            y = y[idx] #根据随机索引打乱数据

            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                # 提取当前批量特征和标签
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                if self.model_type == 'CNN':
                    train_X = train_X.reshape(train_X.shape[0],28,28)
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                # 前向传播
                logits = self.model(train_X)
                # 计算损失，保存
                trn_loss = self.loss_fn(logits, train_y)
                if self.reg != None:
                    trn_loss += self.reg(self.model.layers[0].params) # 只对第一层正则化
                self.train_loss.append(trn_loss)
                
                # 计算评估指标(如准确率)，保存
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # 清除梯度(本程序全连接层的backward()中未累加梯度，不用清零)
                # self.model.clear_grad()

                # 计算最后一层的误差项，再反向传播
                self.model.backward(self.loss_fn.backward())
                if self.reg != None:
                    reg_grads = self.reg.backward(self.model.layers[0].params) # 只对第一层正则化
                    self.model.layers[0].grads['W'] += reg_grads['W']

                # 更新W和b参数(如SGD的step方法)
                self.optimizer.step()

                # 调整学习率
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # 每个batch后，计算验证集评估指标和损失
                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

                # 打印日志：训练集和验证集评估指标、损失
                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

                # 保存最佳模型
                if dev_score > best_score:
                    save_path = os.path.join(save_dir, 'best_model.pickle')
                    self.save_model(save_path)
                    print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                    best_score = dev_score
                self.best_score = best_score

    # 模型评估方法
    def evaluate(self, data_set):
        X, y = data_set
        if self.model_type == 'CNN':
            X = X.reshape(X.shape[0],28,28)
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss


    def save_model(self, save_path):
        self.model.save_model(save_path)