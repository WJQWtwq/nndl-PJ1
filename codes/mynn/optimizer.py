from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        """
        init_lr: 初始学习率
        model: 模型对象
        mu: 动量系数，默认值为 0.9
        """
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}  # 用于存储每一层的动量

        # 初始化每一层的动量为 0，与参数形状一致
        for layer in self.model.layers:
            if layer.optimizable:
                self.velocity[layer] = {key: np.zeros_like(value) for key, value in layer.params.items()}

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    if layer.grads[key] is None:
                        raise ValueError(f"Gradient for {key} in layer {layer} is None. Check the backward implementation.")
                    
                    # 动量更新公式
                    self.velocity[layer][key] = self.mu * self.velocity[layer][key] - self.init_lr * layer.grads[key]
                    layer.params[key] += self.velocity[layer][key]

                    # 权重衰减（如果启用）
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)