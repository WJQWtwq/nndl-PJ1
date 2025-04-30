import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# 测试时选择是MLP还是CNN
model_type = 'CNN'

if model_type == 'CNN':
    model = nn.models.Model_CNN()
elif model_type == 'MLP':
    model = nn.models.Model_MLP()
      
# 修改权重文件位置，测试不同权重文件(详见pdf报告)
model.load_model(r'.\codes\best_models\CNN_big_kernel.pickle')

test_images_path = r'.\codes\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\codes\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

if model_type == 'CNN':
    test_imgs = test_imgs.reshape(test_imgs.shape[0],28,28)

logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))