# -*- coding: gbk -*-
import json
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import random
import os
from sklearn import preprocessing
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

time_start=time.time()
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(1)#15


HSK=[]
total=[]
infile= open("contextual features_test.txt", 'r',encoding='utf-8')

for line in infile:
    if line.split()[0]:
        HSK.append(line.split()[0])
        total.append(line)
infile.close()
x_HSK=[]
ind=[]
for i in range(len(HSK)):
    ci=HSK[i]
    for j in range(len(HSK)):
        if ci==HSK[j]:
            ind.append(j)
            break
for i in range(len(ind)):
    x_HSK.append(list(map(float,total[ind[i]].split()[1:])))


min_max_scaler = preprocessing.MinMaxScaler()
x_HSK= min_max_scaler.fit_transform(x_HSK)
x_HSK=torch.FloatTensor(x_HSK)

lr = 0.0027 # 学习率
gamma=0.85
step_size=100
epochs = 500  # 训练轮数
n_feature = x_HSK.shape[1]  # 输入特征
n_layers=4
# n_hidden = 250 # 隐层节点数
n_hidden1 = 250# 隐层节点数
n_hidden2 = 100 # 隐层节点数
n_hidden3 = 50 # 隐层节点数
n_hidden4 = 20 # 隐层节点数
n_output = 1  # 输出

# n_layers=5
# hidden_size=256
# dropout=0.00041285101332442343
# 2.定义BP神经网络
class BPNetModel1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(BPNetModel1, self).__init__()
        self.hiddden = torch.nn.Linear(n_feature, n_hidden)  # 定义隐层网络
        self.out = torch.nn.Linear(n_hidden, n_output)  # 定义输出层网络

    def forward(self, x):
        x = Fun.relu(self.hiddden(x))  # 隐层激活函数采用relu()函数
        # x=self.hiddden(x)
        # out = Fun.softmax(self.out(x), dim=1)  # 输出层采用softmax函数
        out = self.out(x)
        return out

class BPNetModel2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(BPNetModel2, self).__init__()
        self.hiddden1 = torch.nn.Linear(n_feature, n_hidden1)  # 定义隐层网络
        self.hiddden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # 定义隐层网络
        self.out = torch.nn.Linear(n_hidden2, n_output)  # 定义输出层网络

    def forward(self, x):
        x = Fun.relu(self.hiddden1(x))  # 隐层激活函数采用relu()函数
        x = Fun.relu(self.hiddden2(x))
        # out = Fun.softmax(self.out(x), dim=1)  # 输出层采用softmax函数
        out = self.out(x)
        return out

class BPNetModel3(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2,n_hidden3, n_output):
        super(BPNetModel3, self).__init__()
        self.hiddden1 = torch.nn.Linear(n_feature, n_hidden1)  # 定义隐层网络
        self.hiddden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # 定义隐层网络
        self.hiddden3 = torch.nn.Linear(n_hidden2, n_hidden3)  # 定义隐层网络
        self.out = torch.nn.Linear(n_hidden3, n_output)  # 定义输出层网络

    def forward(self, x):
        x = Fun.relu(self.hiddden1(x))  # 隐层激活函数采用relu()函数
        x = Fun.relu(self.hiddden2(x))
        x = Fun.relu(self.hiddden3(x))
        # out = Fun.softmax(self.out(x), dim=1)  # 输出层采用softmax函数
        out = self.out(x)
        return out

class BPNetModel4(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2,n_hidden3,n_hidden4, n_output):
        super(BPNetModel4, self).__init__()
        self.hiddden1 = torch.nn.Linear(n_feature, n_hidden1)  # 定义隐层网络
        self.hiddden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # 定义隐层网络
        self.hiddden3 = torch.nn.Linear(n_hidden2, n_hidden3)  # 定义隐层网络
        self.hiddden4 = torch.nn.Linear(n_hidden3, n_hidden4)  # 定义隐层网络
        # self.res = torch.nn.Linear(n_feature, n_output)
        self.out = torch.nn.Linear(n_hidden4, n_output)  # 定义输出层网络

    def forward(self, x):
        # x_res = self.res(x)
        x = Fun.relu(self.hiddden1(x))  # 隐层激活函数采用relu()函数
        x = Fun.relu(self.hiddden2(x))
        x = Fun.relu(self.hiddden3(x))
        x = Fun.relu(self.hiddden4(x))
        # out = Fun.softmax(self.out(x), dim=1)  # 输出层采用softmax函数
        out = self.out(x)
        # out = Fun.relu(self.out(x))
        return out

if n_layers==1:
    net = BPNetModel1(n_feature=n_feature, n_hidden1=n_hidden1, n_output=n_output)  # 调用网络
if n_layers == 2:
    net = BPNetModel2(n_feature=n_feature, n_hidden1=n_hidden1,n_hidden2=n_hidden2, n_output=n_output)  # 调用网络
if n_layers == 3:
    net = BPNetModel3(n_feature=n_feature, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_hidden3=n_hidden3,n_output=n_output)  # 调用网络
if n_layers == 4:
    net = BPNetModel4(n_feature=n_feature, n_hidden1=n_hidden1,n_hidden2=n_hidden2,n_hidden3=n_hidden3,n_hidden4=n_hidden4, n_output=n_output)  # 调用网络

state_dict = torch.load('model.pth')
net.load_state_dict(state_dict['model'])
y_HSK = net(x_HSK)
y_HSK=(y_HSK+0.1)*5
y_HSK=y_HSK.reshape(-1).detach().numpy()

df=pd.DataFrame({'Written Chinese Words':HSK,'Elegant-formality':y_HSK})
with pd.ExcelWriter("test_result.xlsx") as writer:
    df.to_excel(writer, sheet_name='Sheet1', index=False)

time_end=time.time()
print('程序运行时间:',time_end-time_start)