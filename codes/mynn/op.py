from abc import abstractmethod
from tkinter import W
import numpy as np

# 抽象基类
class Layer():
    def __init__(self) -> None:
        self.optimizable = True  # 是否可优化，更新权重
    
    @abstractmethod # 装饰器，表示子类必须实现该方法
    def forward():
        pass

    @abstractmethod
    def backward():
        pass

# 全连接层，继承Layer (如optimizable参数)
class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.

    1.in_dim输入维度，out_dim输出维度
    2.initialize_method权重初始化方法：降低发生梯度爆炸和梯度消失的风险
        (1)非两端饱和->爆炸或消失；两端饱和（sigmoid）->消失；
        (2)常数/均匀分布初始化（不好）；正态初始化；Xavier初始化；He初始化
    3.weight_decay是否使用权重衰减，weight_decay_lambda权重衰减强度
    """

    # 单线性层初始化
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim)) # 初始化权重矩阵：in_dim×out_dim
        self.b = initialize_method(size=(1, out_dim))      # 初始化偏置向量：1×out_dim
        self.grads = {'W' : None, 'b' : None}              # 初始化权重、偏置梯度为None
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b} # 

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    # __call__:可直接调用实例对象来执行 forward
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    # 单线性层前传
    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        batch_size:常2^n，小批量梯度下降法！！！
        """
        self.input = X # 记录输入，反向传播时使用
        return np.dot(X,self.params['W']) + self.params['b']
        # 下一层的输入
        # X（batch_size,in_dim）dot self.W(in_dim,out_dim) + self.b(1,out_dim，广播相加)

    # 单线性层反传
    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = grad.shape[0]
        self.grads['W'] = np.dot(self.input.T, grad) / batch_size #W平均梯度(P97)，idim×bsize dot bsize×odim
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size #b平均梯度(P98)，对bsize维求和，保持维度
        
        #传给上一层(i->上层)，(bsize,odim) @ (odim,idim) = (bsize,idim)
        #grad2prev=grad dot W dot diag(fl'(Z_l)),ReLU diag(fl'(Z_l))=I_Ml,省略
        grad2prev = np.dot(grad, self.params['W'].T)
        return grad2prev
    
    # 清除梯度
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

# 二维卷积层，继承Layer(如optimizable)，下一层为全连接层
# 暂时只考虑单通道
class Conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    1.in_channels输入通道数，out_channels输出通道数，kernel_size卷积核大小
    2.stride步长，padding填充
    """
    def __init__(self, in_channels=1, out_channels=1, kernel_size=5, kernel_num=3, stride=1, padding=0, 
                 initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8, next_linear=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_num =kernel_num
        self.stride = stride
        self.padding = padding
        self.initialize_method = initialize_method
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.next_linear = next_linear
        
        self.W = initialize_method(size=(kernel_num, kernel_size, kernel_size))
        self.b = initialize_method(size=(kernel_num,))
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, H, W]
        output : [batch, out_dim]
        no padding
        """
        self.input = X #记录输入，反向传播时使用
        batch_size, H, W = X.shape 
        num, k, _ = self.params['W'].shape
        out_H = (H-k)//self.stride + 1
        out_W = (W-k)//self.stride + 1
        output = np.zeros((batch_size, self.kernel_num, out_H, out_W))
        self.out_H = out_H
        self.out_W = out_W

        # 二维互相关
        for i in range(out_H):
            for j in range(out_W):
                h_start = i*self.stride
                h_end = h_start + k
                w_start = j*self.stride
                w_end = w_start + k

                X_slice = X[:,h_start:h_end,w_start:w_end]
                for s in range(self.kernel_num): 
                    for t in range(batch_size):     
                        output[t,s,i,j] = np.sum(X_slice[t]*self.params['W'][s]) + self.params['b'][s]
                # X_slice*self.W:(bsize,ichannels,k,k),多输入通道全部累加
        if self.next_linear:
            output = output.reshape(batch_size, self.kernel_num*out_H*out_W)
        return output
    
    def backward(self, grads):
        """
        grads : [batch_size, out_dim]
        output : [batch_size, kernel_num, H, W]
        """ 
        batch_size = grads.shape[0] 
        if self.next_linear: #若下一层为全连接层，变形
            grads = grads.reshape(batch_size, self.kernel_num, self.out_H, self.out_W) 
        _, in_H, in_W = self.input.shape
        # 注意H、W是什么！偏置项与输入卷积！
        # H, W = (in_H-self.out_H)//self.stride + 1, (in_W-self.out_W)//self.stride + 1      
        k = self.kernel_size

        dW = np.zeros_like(self.params['W'])
        db = np.zeros_like(self.params['b'])
        dX = np.zeros_like(self.input)

        # grads_pad = np.pad(grads,((k-1,k-1),(k-1,k-1)),'constant',constant_values = (0,0))
        # k_rot180 = np.rot90(np.rot90(self.params['W'])) 
        for i in range(k):
            for j in range(k):
                h_start = i*self.stride
                h_end = h_start + self.out_H
                w_start = j*self.stride
                w_end = w_start + self.out_W

                X_slice = self.input[:,h_start:h_end,w_start:w_end]
                #grads_slice = grads[:,:,]

                for s in range(self.kernel_num):
                    for t in range(batch_size):
                        dW[s,i,j] += np.sum(X_slice[t]*grads[t,s,:,:])
                                  
                #dX[:,:,] = 
                    # 当前卷积层为首层，暂不考虑反传误差项
                    # dX[:,h_start:h_end,w_start:w_end] += self.params['W']*grads_slice  
        for s in range(self.kernel_num):
            db[s] += np.sum(grads[:,s,:,:])
        self.grads['W'] = dW/batch_size
        self.grads['b'] = db/batch_size
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class Max_Pooling(Layer):
    def __init__(self, pooling_size=2):
        self.optimizable = False
        self.pooling_size = pooling_size

    def __call__(self,X):
        return self.foward(X)
    
    def forward(self,X):
        '''
        input:[batch_size, in_channels, in_H, in_W]
        output:[batch_size, in_channels, out_H, out_W]
        '''
        self.input = X
        batch_size, in_channels, in_H, in_W = X.shape
        out_H ,out_W = in_H//2, in_W//2
        ps = self.pooling_size
        output = np.zeros((batch_size, in_channels, out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                output[:,:,i,j] = np.max(X[:,:, ps*i:ps*(i+1), ps*j:ps*(j+1)])

        self.output = output
        return output
    
    def backward(self,X):
        '''
        input:[batch_size, in_channels, out_H, out_W]
        output:[batch_size, in_channels, in_H, in_W]
        '''
        batch_size, in_channels, out_H, out_W = X.shape
        ps = self.pooling_size
        output = np.zeros_like(self.input)

        # 最大汇聚：最大值位置传递X值，其余置零
        for i in range(out_H):
            for j in range(out_W):
                output[:,:,ps*i:ps*(i+1), ps*j:ps*(j+1)] = np.where(self.output[:,:,i,j]==output, X[:,:,i,j], 0)
        return output

class ReLU(Layer):
    """
    An activation layer.
    """
    # 单激活层初始化
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    # 前向传播到ReLU层时调用
    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)  #ReLU
        return output
    
    # 反向传播到ReLU层时调用
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)  #ReLU
        return output

class Leaky_ReLU(Layer):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.optimizable = False
        self.input = None
        self.gamma = gamma
    
    def __call__(self,X):
        return self.forward(X)
    
    def forward(self,X):
        self.input = X
        output = np.where(X<0, self.gamma*X, X)
        return output
    
    def backward(self,grads):
        assert grads.shape == self.input.shape
        output = np.where(self.input<0, self.gamma*grads, grads)
        return output

# 交叉熵损失层
class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    # 算交叉熵损失，前向传播时调用
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        batch_size = predicts.shape[0]
        if self.has_softmax: 
            predicts = softmax(predicts) # 用softmax函数赋予预测每一维的概率
        self.predicts = predicts
        self.labels = labels

        # one-hot式交叉熵P29
        loss = -np.sum(np.log(predicts[np.arange(batch_size), labels]+1e-7)) / batch_size
        return loss
    
    # 反向传播时调用，计算最后一层误差项δ_L，定义见nndl书P97
    def backward(self): 
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        batch_size = self.predicts.shape[0]
        self.grads = self.predicts.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1 # 求交叉熵损失函数导数(减y_i即减1)
        self.grads /= batch_size # 平均导数

        return self.grads

    # 取消softmax,使用hardmax
    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Regularization can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, weight_decay_lambda=1e-4):
        """
        weight_decay_lambda: 控制正则化强度的超参数
        """
        super().__init__()
        self.weight_decay_lambda = weight_decay_lambda
        self.optimizable = False  # L2 正则化本身不需要优化

    def __call__(self, params):
        return self.forward(params)

    def forward(self, params):
        """
        计算 L2 正则化项的值
        params: 模型的参数字典，例如 {'W': 权重矩阵, 'b': 偏置向量}
        """
        l2_loss = 0
        for key, value in params.items():
            if key == 'W':  # 只对权重矩阵 W 应用 L2 正则化
                l2_loss += 0.5 * self.weight_decay_lambda * np.sum(value ** 2)
        return l2_loss

    def backward(self, params):
        """
        计算 L2 正则化项对参数的梯度
        params: 模型的参数字典，例如 {'W': 权重矩阵, 'b': 偏置向量}
        """
        l2_grads = {}
        for key, value in params.items():
            if key == 'W':  # 只对权重矩阵 W 应用梯度
                l2_grads[key] = self.weight_decay_lambda * value
        return l2_grads
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

if __name__ == '__main__':
    # linear = Linear(in_dim=3, out_dim=2)
    # X = np.array([[1, 2, 3], [4, 5, 6]])
    # output = linear.forward(X)
    # print("Linear Forward Output:", output)

    # grad = np.array([[0.1, 0.2], [0.3, 0.4]])
    # grad2prev = linear.backward(grad)
    # print("Linear Backward Grad to Previous Layer:", grad2prev)
    # print("Linear Grads for W:", linear.grads['W'])
    # print("Linear Grads for b:", linear.grads['b'])

    # 测试ReLU层
    # relu = ReLU()
    # X = np.array([[1,2],[3,4]])
    # output = relu.forward(X)
    # print("ReLU Forward Output:", output)

    # grad = np.array([[0.1, 0.2], [0.3, 0.4]])
    # grad_to_prev = relu.backward(grad)
    # print("ReLU Backward Grad to Previous Layer:", grad_to_prev)

    # 测试MultiCrossEntropyLoss层
    loss_layer = MultiCrossEntropyLoss()
    predicts = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([1, 0])
    loss = loss_layer.forward(predicts, labels)
    print("MultiCrossEntropyLoss Forward Loss:", loss)

    grads = loss_layer.backward()
    print("MultiCrossEntropyLoss Backward Grads:\n", grads)