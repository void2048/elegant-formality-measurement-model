# -*- coding: gbk -*-
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import random
import os
from sklearn.metrics import r2_score
from sklearn import preprocessing
import optuna

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
set_seed(111)
set_seed(15)#15
df=pd.read_excel("lexical features.xlsx",sheet_name='庄雅度')
#semantic features
cixiangliang=[]
total=[]
infile= open("semantic features.txt", 'r', encoding='utf-8')
for line in infile:
    cixiangliang.append(line.split()[0])
    total.append(line)
infile.close()
x=[]
y=[]
ci_list=[]
ind=[]
for i in range(len(df)):
    ci=df.iloc[i]['词语']
    for j in range(len(cixiangliang)):
        if ci==cixiangliang[j]:
            ind.append(j)
            ci_list.append(ci)
            y.append(df.iloc[i]['庄雅度'])
            break
for i in range(len(ind)):
    x.append(list(map(float,total[ind[i]].split()[1:])))

#contextual features
cixiangliang=[]
total=[]
infile= open("contextual features.txt", 'r', encoding='utf-8')
for line in infile:
    cixiangliang.append(line.split()[0])
    total.append(line)
infile.close()
x_yujingtezheng=[]
ind=[]
for i in range(len(ci_list)):
    ci=ci_list[i]
    for j in range(len(cixiangliang)):
        if ci==cixiangliang[j]:
            ind.append(j)
            break
for i in range(len(ind)):
    x_yujingtezheng.append(list(map(float,total[ind[i]].split()[1:])))

#lexical features
x_cichang=[]
x_yinjieshu=[]
x_cixing=[]
cixing_list=['x', 'v', 'n', 'u', 'd', 'nt', 'p', 'a', 'm', '1', 'r', 'e', 'c', 'q', 'nh', 'nl', 'ns', 'nd', 'k', 'mq', 'vu', 'o']
x_cizhui=[]
cizhui_list=['无','有']
ind=[]

for i in range(len(df)):
    ci=df.iloc[i]['词语']
    for j in range(len(ci_list)):
        if ci==ci_list[j]:
            cixing = df.iloc[i]['词性'].split('/')[1]
            cizhui=df.iloc[i]['词缀']
            ind.append(i)
            x_cichang.append([int(df.iloc[i]['词长'])])
            x_yinjieshu.append([int(df.iloc[i]['音节数'])])
            one_hot=[0]*len(cixing_list)
            one_hot[cixing_list.index(cixing)]=1
            x_cixing.append(one_hot)
            one_hot = [0] * len(cizhui_list)
            one_hot[cizhui_list.index(cizhui)] = 1
            x_cizhui.append(one_hot)
            break

#相对词频
# infile= open("语料库和词频统计/词频统计/书面语词频统计#20240326182045.txt", 'r',encoding='gbk')
# shumian_ci=[]
# total_shumian=[]
# for line in infile:
#     shumian_ci.append(line.split()[1])
#     total_shumian.append(line)
# infile.close()
#
# infile= open("语料库和词频统计/词频统计/口语词频统计#20240325213648.txt", 'r',encoding='gbk')
# kou_ci=[]
# total_kou=[]
# for line in infile:
#     kou_ci.append(line.split()[1])
#     total_kou.append(line)
# infile.close()
#
# shumiancipin=[0]*len(ci_list)
# for i in range(len(ci_list)):
#     ci=ci_list[i]
#     for j in range(len(shumian_ci)):
#         if shumian_ci[j]==ci:
#             shumiancipin[i]=float(total_shumian[j].split()[3])
# # print(shumiancipin)
#
# koucipin=[0]*len(ci_list)
# for i in range(len(ci_list)):
#     ci=ci_list[i]
#     for j in range(len(kou_ci)):
#         if kou_ci[j]==ci:
#             koucipin[i]=float(total_kou[j].split()[3])
# # print(koucipin)
# # os.system('pause')
# xiangduicipin=[-1]*len(ci_list)
# for i in range(len(ci_list)):
#     if koucipin[i]!=0:
#         xiangduicipin[i]=np.log(shumiancipin[i]/koucipin[i])
# xiangduicipin_max=max(xiangduicipin)
# print(xiangduicipin)
# f=open('./相对词频.txt','w',encoding='utf-8')
# for i in range(len(ci_list)):
#     if xiangduicipin[i]==-1:
#         xiangduicipin[i]=xiangduicipin_max
#     f.write(ci_list[i]+' %f\n'%(xiangduicipin[i]))
#     xiangduicipin[i]=[xiangduicipin[i]]
# f.close()
# print(xiangduicipin)


f=open('../相对词频.txt', 'r', encoding='utf-8')
x_xiangduicipin=[]
for line in f:
    x_xiangduicipin.append([float(line.split()[1])])
f.close()
# print(x_xiangduicipin)

# #IDF逆向文档频率
# files = os.listdir('语料库和词频统计')[:5]  # 得到文件夹下的所有文件名称
# txts_total=[]
#
# for corpus_type in files:
#     txts_names=os.listdir('语料库和词频统计/'+corpus_type)
#     # print(txts_names[-1])
#     read_error_num = 0
#     for txts_name in txts_names:
#         # print('语料库和词频统计/'+corpus_type+'/'+txts_name)
#         f=open('语料库和词频统计/'+corpus_type+'/'+txts_name, "r", encoding='utf-8')
#         # with open('语料库和词频统计/'+corpus_type+'/'+txts_name, "r", encoding='utf-8') as f:  # 打开文件
#         try:
#             txt = f.read()  # 读取文件
#         except:
#             # print('not utf_8')
#             try:
#                 f = open('语料库和词频统计/' + corpus_type + '/' + txts_name, "r", encoding='gbk')
#                 txt = f.read()  # 读取文件
#             except:
#                 # print('not gbk or utf-8')
#                 read_error_num+=1
#                 continue
#         txts_total.append(txt)
#     print('read error num:',read_error_num)
# # temp=[]
# for i in range(len(txts_total)):
#     txts_total[i]=txts_total[i].split()
#     # temp.append(txts_total[i].split())
# txts_total_num=len(txts_total)
# print('总文本数:',len(txts_total))
# # txts_total=temp
# x_IDF=[]
# c_flag=0
# f=open('逆向文档频率.txt','w',encoding='utf_8')
# for k in range(len(ci_list)):
#     ci=ci_list[k]
#     print('正在处理第{}个词语:'.format(k), ci)
#     # if ci=='宁静':
#     #     c_flag=1
#     #     continue
#     # if c_flag==0:
#     #     continue
#     dft=1
#     for i in range(len(txts_total)):
#         flag=0
#         for j in range(len(txts_total[i])):
#             s = txts_total[i][j]
#             s = s.split('/')[0]
#             if s==ci:
#                 flag=1
#                 break
#         if flag==1:
#             dft+=1
#     f.write(ci_list[k] + ' %f\n' % (np.log(txts_total_num/dft)))
#     x_IDF.append([np.log(txts_total_num/dft)])
#     # break
# f.close()
#
f=open('../逆向文档频率.txt', 'r', encoding='utf-8')
x_IDF=[]
for line in f:
    x_IDF.append([float(line.split()[1])])
f.close()
# print('x_IDF:',x_IDF)

#词频-逆文档频率TF-IDF
x_TF_IDF=[]
for i in range(len(df)):
    ci=df.iloc[i]['词语']
    for j in range(len(ci_list)):
        if ci==ci_list[j]:
            x_TF_IDF.append([x_IDF[j][0]*df.iloc[i]['频率']])
#特征拼接
x_yufatezheng=[0]*len(x)
for i in range(len(x)):
    x_yufatezheng[i]=x_cichang[i]+x_yinjieshu[i]+x_cixing[i]+x_cizhui[i]+x_xiangduicipin[i]+x_IDF[i]+x_TF_IDF[i]



for i in range(len(x)):
    # x[i]=x[i]+x_yujingtezheng[i]+x_yufatezheng[i]
    # x[i] = x[i] + x_yujingtezheng[i]
    # x[i] = x[i]
    x[i] = x_yujingtezheng[i]
    # x[i] = x_yufatezheng[i]
    # x[i] = x_yujingtezheng[i]+x_yufatezheng[i]
    # x[i] = x_yufatezheng[i]+x[i]
    # x[i] = x_yujingtezheng[i] + x[i]
# os.system('pause')

rind=[i for i in range(len(x))]
random.shuffle(rind)

min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)


x=torch.FloatTensor(x)
y=torch.FloatTensor(y)/5-0.1


class BPNetModel1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_output):
        super(BPNetModel1, self).__init__()
        self.hiddden = torch.nn.Linear(n_feature, n_hidden1)  # 定义隐层网络
        self.out = torch.nn.Linear(n_hidden1, n_output)  # 定义输出层网络

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
        self.out = torch.nn.Linear(n_hidden4, n_output)  # 定义输出层网络

    def forward(self, x):
        x = Fun.relu(self.hiddden1(x))  # 隐层激活函数采用relu()函数
        x = Fun.relu(self.hiddden2(x))
        x = Fun.relu(self.hiddden3(x))
        x = Fun.relu(self.hiddden4(x))
        # out = Fun.softmax(self.out(x), dim=1)  # 输出层采用softmax函数
        out = self.out(x)
        return out

class BPNetModel5(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2,n_hidden3,n_hidden4,n_hidden5, n_output):
        super(BPNetModel4, self).__init__()
        self.hiddden1 = torch.nn.Linear(n_feature, n_hidden1)  # 定义隐层网络
        self.hiddden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # 定义隐层网络
        self.hiddden3 = torch.nn.Linear(n_hidden2, n_hidden3)  # 定义隐层网络
        self.hiddden4 = torch.nn.Linear(n_hidden3, n_hidden4)  # 定义隐层网络
        self.hiddden5 = torch.nn.Linear(n_hidden4, n_hidden5)  # 定义隐层网络
        self.out = torch.nn.Linear(n_hidden5, n_output)  # 定义输出层网络

    def forward(self, x):
        x = Fun.relu(self.hiddden1(x))  # 隐层激活函数采用relu()函数
        x = Fun.relu(self.hiddden2(x))
        x = Fun.relu(self.hiddden3(x))
        x = Fun.relu(self.hiddden4(x))
        x = Fun.relu(self.hiddden5(x))
        # out = Fun.softmax(self.out(x), dim=1)  # 输出层采用softmax函数
        out = self.out(x)
        return out

class BPNetModel6(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2,n_hidden3,n_hidden4,n_hidden5,n_hidden6, n_output):
        super(BPNetModel4, self).__init__()
        self.hiddden1 = torch.nn.Linear(n_feature, n_hidden1)  # 定义隐层网络
        self.hiddden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # 定义隐层网络
        self.hiddden3 = torch.nn.Linear(n_hidden2, n_hidden3)  # 定义隐层网络
        self.hiddden4 = torch.nn.Linear(n_hidden3, n_hidden4)  # 定义隐层网络
        self.hiddden5 = torch.nn.Linear(n_hidden4, n_hidden5)  # 定义隐层网络
        self.out = torch.nn.Linear(n_hidden5, n_output)  # 定义输出层网络

    def forward(self, x):
        x = Fun.relu(self.hiddden1(x))  # 隐层激活函数采用relu()函数
        x = Fun.relu(self.hiddden2(x))
        x = Fun.relu(self.hiddden3(x))
        x = Fun.relu(self.hiddden4(x))
        x = Fun.relu(self.hiddden5(x))
        # out = Fun.softmax(self.out(x), dim=1)  # 输出层采用softmax函数
        out = self.out(x)
        return out


def train(x_train,y_train,x_test,y_test,lr,gamma,step_size,epochs,n_feature,n_layers,n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_output):
    if n_layers==1:
        net = BPNetModel1(n_feature=n_feature, n_hidden1=n_hidden1, n_output=n_output)  # 调用网络
    if n_layers == 2:
        net = BPNetModel2(n_feature=n_feature, n_hidden1=n_hidden1,n_hidden2=n_hidden2, n_output=n_output)  # 调用网络
    if n_layers == 3:
        net = BPNetModel3(n_feature=n_feature, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_hidden3=n_hidden3,n_output=n_output)  # 调用网络
    if n_layers == 4:
        net = BPNetModel4(n_feature=n_feature, n_hidden1=n_hidden1,n_hidden2=n_hidden2,n_hidden3=n_hidden3,n_hidden4=n_hidden4, n_output=n_output)  # 调用网络

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 使用Adam优化器，并设置学习率
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=step_size)
    loss_fun = torch.nn.MSELoss()  # 对于多分类一般使用交叉熵损失函数
    L1_fun = torch.nn.L1Loss()
    loss_steps = []  # 构造一个array([ 0., 0., 0., 0., 0.])里面有epochs个0
    accuracy_steps = []
    R2=[]
    R2_last_n_list = []
    max_R2 = float('-inf')
    optimizer.zero_grad()
    for epoch in range(epochs):
        y_pred = net(x_train)  # 前向传播
        loss = loss_fun(y_pred.reshape(-1), y_train)  # 预测值和真实值对比
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新梯度
        lr_scheduler.step()
        loss_steps.append(loss.item())  # 保存loss
        running_loss = loss.item()
        if epoch%200==0:
            print(f"第{epoch}次训练，loss={running_loss}".format(epoch, running_loss))
        with torch.no_grad():  # 下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
            y_pred = net(x_test)
            L1loss = torch.abs(y_pred-y_test.reshape(-1, 1))
            accuracy_steps.append(L1loss.mean())
            R2.append(r2_score(y_test.reshape(-1).detach().numpy(),y_pred.reshape(-1).detach().numpy()))
        max_R2 = max(max_R2, R2[-1])
        R2_last_n_list.append(R2[-1])
        if len(R2_last_n_list) > 30:
            R2_last_n_list.pop(0)
        max_R2_over_last_n_epochs = max(R2_last_n_list)
        # 提前停止条件：如果最近若干个epoch的性能较最优值都没有提高，则停止训练
        if max_R2 > max_R2_over_last_n_epochs:
            break
    print('L1loss_std:',np.std(L1loss.detach().numpy()))
    print(torch.sum(L1loss<0.2)/len(y_test))
    print(min(accuracy_steps))
    print('R2:',R2[-1])
    return accuracy_steps[-1],R2[-1],(torch.sum(L1loss<0.2)/len(y_test)).detach().numpy(),(torch.sum(L1loss<0.1)/len(y_test)).detach().numpy()

now_trial=1
def objective(trial):
    global now_trial
    print('正在进行第%d次优化'%now_trial)
    now_trial+=1
    lr = trial.suggest_float('lr', 1e-5, 1e-1,log=True) # 学习率
    gamma = trial.suggest_float('gamma', 0.5, 1.0)
    step_size = trial.suggest_categorical('step_size', [50, 100, 200, 300,400])
    epochs = 5000  # 训练轮数
    n_feature = x.shape[1]  # 输入特征
    n_layers=trial.suggest_categorical('n_layers', [1,2,3,4])
    # n_hidden1 = trial.suggest_int('n_hidden1', 1, 300)# 隐层节点数
    # n_hidden2 = trial.suggest_int('n_hidden2', 1, 200)# 隐层节点数
    # n_hidden3 = trial.suggest_int('n_hidden3', 1, 200)# 隐层节点数
    # n_hidden4 = trial.suggest_int('n_hidden4', 1, 100)# 隐层节点数
    n_hidden1 = trial.suggest_categorical('n_hidden1', [150,200,250,300])  # 隐层节点数
    n_hidden2 = trial.suggest_categorical('n_hidden2', [80,100,120,150])  # 隐层节点数
    n_hidden3 = trial.suggest_categorical('n_hidden3', [30,40,50])  # 隐层节点数
    n_hidden4 = trial.suggest_categorical('n_hidden4', [5,10,20])  # 隐层节点数
    n_output = 1  # 输出
    L1_list=[]
    R2_list=[]
    less_than02_list=[]
    less_than01_list=[]
    cut_num=int(len(x)/5)
    for i in range(5):
        x_test=x.index_select(0,torch.tensor(rind[i*cut_num:(i+1)*cut_num]))
        y_test=y.index_select(0,torch.tensor(rind[i*cut_num:(i+1)*cut_num]))
        x_train=x.index_select(0,torch.tensor(rind[:i*cut_num]+rind[(i+1)*cut_num:]))
        y_train=y.index_select(0,torch.tensor(rind[:i*cut_num]+rind[(i+1)*cut_num:]))
        L1,R2,less_than02,less_than01=train(x_train, y_train, x_test, y_test,lr,gamma,step_size,epochs,n_feature,n_layers,n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_output)
        L1_list.append(L1)
        R2_list.append(R2)
        less_than02_list.append(less_than02)
        less_than01_list.append(less_than01)
    return np.mean(R2_list)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print('Best trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))


time_end=time.time()
print('程序运行时间:',time_end-time_start)