from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1):
        """
        初始化 MultiStepLR 调度器。
        
        :param optimizer: 优化器对象，例如 SGD。
        :param milestones: 一个列表，包含学习率调整的迭代次数。
        :param gamma: 学习率衰减因子，默认为 0.1。
        """
        self.optimizer = optimizer
        self.milestones = sorted(milestones)  # 确保里程碑按升序排列
        self.gamma = gamma
        self.current_lr = optimizer.init_lr
        self.current_step = 0

    def step(self):
        """
        调整学习率。
        """
        self.current_step += 1
        if self.current_step in self.milestones:
            self.current_lr *= self.gamma
            self.optimizer.init_lr = self.current_lr

class ExponentialLR(scheduler):
    pass