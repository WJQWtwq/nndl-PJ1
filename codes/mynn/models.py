from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    # 整个多层网络的初始化
    # size_list每一层的神经元数量,act_func激活函数，lambda_list权重衰减系数
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []  # 存储模型的所有层
            for i in range(len(size_list) - 1): # 遍历 size_list，构建每一层
                # 创建一个线性层
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer) #添加该层

                # 如果不是最后一层，则将激活层添加到 self.layers
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    # 整个多层网络的前传
    def forward(self, X):
        # 确保模型已经初始化
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    # 整个多层网络的反传
    def backward(self, loss_grad):
        grads = loss_grad # 初始化梯度为损失函数梯度
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    # 加载模型，测试训练好的模型时用
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)

    # 保存训练好的模型 
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:  # 只保存可优化层的参数
                # 保存权重、偏置、权重衰减设置
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, kernel_size=5, 
                 kernel_num=3, in_channels=None, out_channels=None):  
        self.size_list = size_list
        self.act_func = act_func
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num

        if size_list is not None and act_func is not None:
            self.layers = []  # 存储模型的所有层
            # 创建一个卷积层
            layer = Conv2D(kernel_size=kernel_size,kernel_num=kernel_num,next_linear=True)
            if lambda_list is not None:
                layer.weight_decay = True
                layer.weight_decay_lambda = lambda_list[0]
            if act_func[0] == 'Logistic':
                raise NotImplementedError
            elif act_func[0] == 'ReLU':
                layer_f = ReLU()
            elif act_func[0] == 'Leaky_ReLU':
                layer_f = Leaky_ReLU()
            self.layers.append(layer) #添加该层

            # 将激活层添加到 self.layers
            if act_func[0] != 'None':
                self.layers.append(layer_f)

            # 添加线性层
            self.size_list[0] = kernel_num*((size_list[0]-kernel_size+1) ** 2)
            for i in range(len(size_list) - 1): # 遍历 size_list，构建每一层
                # 创建一个线性层
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func[i+1] == 'Logistic':
                    raise NotImplementedError
                elif act_func[i+1] == 'ReLU':
                    layer_f = ReLU()
                elif act_func[i+1] == 'Leaky_ReLU':
                    layer_f = Leaky_ReLU()
                self.layers.append(layer) #添加该层

                # 如果不是最后一层，则将激活层添加到 self.layers(最后一层加softmax)
                if i < len(size_list) - 2 and act_func[i+1] != 'None':
                    self.layers.append(layer_f)

                # 输出层不加激活层！！！


    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # 确保模型已经初始化
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad # 初始化梯度为损失函数梯度
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    # 加载模型，测试训练好的模型时用
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.kernnel_size = param_list[2]

        self.layers = []
        
        # 加载卷积层
        layer = Conv2D(kernel_size=param_list[2],kernel_num=3,next_linear=True)
        layer.W = param_list[3]['W']
        layer.b = param_list[3]['b']
        layer.params['W'] = layer.W
        layer.params['b'] = layer.b
        layer.weight_decay = param_list[3]['weight_decay']
        layer.weight_decay_lambda = param_list[3]['lambda']
        if self.act_func == 'Logistic':
            raise NotImplemented
        elif self.act_func == 'ReLU':
            layer_f = ReLU()
        elif self.act_func == 'Leaky_ReLU':
            layer_f = Leaky_ReLU()
        self.layers.append(layer)
        #if  self.act_func[0] != 'None':
        #    self.layers.append(layer_f)
        
        # 加载线性层和输出层
        for i in range(len(self.size_list)-1):
            layer = Linear(in_dim=self.size_list[i],out_dim=self.size_list[i+1])
            layer.W = param_list[i+4]['W']
            layer.b = param_list[i+4]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i+4]['weight_decay']
            layer.weight_decay_lambda = param_list[i+4]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            elif self.act_func == 'Leaky_ReLU':
                layer_f = Leaky_ReLU()
            self.layers.append(layer)
            if i < len(self.size_list)-2 : #and self.act_func[i+1] != 'None':
                self.layers.append(layer_f)

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func, self.kernel_size]
        for layer in self.layers:
            if layer.optimizable:  # 只保存可优化层的参数
                # 保存权重、偏置、权重衰减设置
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)