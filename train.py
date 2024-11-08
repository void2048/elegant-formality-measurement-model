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
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegression

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
set_seed(15)#15
df=pd.read_excel("lexical features.xlsx",sheet_name='ׯ�Ŷ�')
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
    ci=df.iloc[i]['����']
    for j in range(len(cixiangliang)):
        if ci==cixiangliang[j]:
            ind.append(j)
            ci_list.append(ci)
            y.append(df.iloc[i]['ׯ�Ŷ�'])
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
cizhui_list=['��','��']
ind=[]

for i in range(len(df)):
    ci=df.iloc[i]['����']
    for j in range(len(ci_list)):
        if ci==ci_list[j]:
            cixing = df.iloc[i]['����'].split('/')[1]
            cizhui=df.iloc[i]['��׺']
            ind.append(i)
            x_cichang.append([int(df.iloc[i]['�ʳ�'])])
            x_yinjieshu.append([int(df.iloc[i]['������'])])
            one_hot=[0]*len(cixing_list)
            one_hot[cixing_list.index(cixing)]=1
            x_cixing.append(one_hot)
            one_hot = [0] * len(cizhui_list)
            one_hot[cizhui_list.index(cizhui)] = 1
            x_cizhui.append(one_hot)
            break

#��Դ�Ƶ
# infile= open("���Ͽ�ʹ�Ƶͳ��/��Ƶͳ��/�������Ƶͳ��#20240326182045.txt", 'r',encoding='gbk')
# shumian_ci=[]
# total_shumian=[]
# for line in infile:
#     shumian_ci.append(line.split()[1])
#     total_shumian.append(line)
# infile.close()
#
# infile= open("���Ͽ�ʹ�Ƶͳ��/��Ƶͳ��/�����Ƶͳ��#20240325213648.txt", 'r',encoding='gbk')
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
# f=open('./��Դ�Ƶ.txt','w',encoding='utf-8')
# for i in range(len(ci_list)):
#     if xiangduicipin[i]==-1:
#         xiangduicipin[i]=xiangduicipin_max
#     f.write(ci_list[i]+' %f\n'%(xiangduicipin[i]))
#     xiangduicipin[i]=[xiangduicipin[i]]
# f.close()
# print(xiangduicipin)


f=open('../��Դ�Ƶ.txt', 'r', encoding='utf-8')
x_xiangduicipin=[]
for line in f:
    x_xiangduicipin.append([float(line.split()[1])])
f.close()
# print(x_xiangduicipin)

# #IDF�����ĵ�Ƶ��
# files = os.listdir('���Ͽ�ʹ�Ƶͳ��')[:5]  # �õ��ļ����µ������ļ�����
# txts_total=[]
#
# for corpus_type in files:
#     txts_names=os.listdir('���Ͽ�ʹ�Ƶͳ��/'+corpus_type)
#     # print(txts_names[-1])
#     read_error_num = 0
#     for txts_name in txts_names:
#         # print('���Ͽ�ʹ�Ƶͳ��/'+corpus_type+'/'+txts_name)
#         f=open('���Ͽ�ʹ�Ƶͳ��/'+corpus_type+'/'+txts_name, "r", encoding='utf-8')
#         # with open('���Ͽ�ʹ�Ƶͳ��/'+corpus_type+'/'+txts_name, "r", encoding='utf-8') as f:  # ���ļ�
#         try:
#             txt = f.read()  # ��ȡ�ļ�
#         except:
#             # print('not utf_8')
#             try:
#                 f = open('���Ͽ�ʹ�Ƶͳ��/' + corpus_type + '/' + txts_name, "r", encoding='gbk')
#                 txt = f.read()  # ��ȡ�ļ�
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
# print('���ı���:',len(txts_total))
# # txts_total=temp
# x_IDF=[]
# c_flag=0
# f=open('�����ĵ�Ƶ��.txt','w',encoding='utf_8')
# for k in range(len(ci_list)):
#     ci=ci_list[k]
#     print('���ڴ����{}������:'.format(k), ci)
#     # if ci=='����':
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
f=open('../�����ĵ�Ƶ��.txt', 'r', encoding='utf-8')
x_IDF=[]
for line in f:
    x_IDF.append([float(line.split()[1])])
f.close()
# print('x_IDF:',x_IDF)

#��Ƶ-���ĵ�Ƶ��TF-IDF
x_TF_IDF=[]
for i in range(len(df)):
    ci=df.iloc[i]['����']
    for j in range(len(ci_list)):
        if ci==ci_list[j]:
            x_TF_IDF.append([x_IDF[j][0]*df.iloc[i]['Ƶ��']])
#����ƴ��
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


lr = 0.0027 # ѧϰ��
gamma=0.85
step_size=100
epochs = 5000  # ѵ������
n_feature = x.shape[1]  # ��������
n_layers=4
# n_hidden = 250 # ����ڵ���
n_hidden1 = 250# ����ڵ���
n_hidden2 = 100 # ����ڵ���
n_hidden3 = 50 # ����ڵ���
n_hidden4 = 20 # ����ڵ���
n_output = 1  # ���

# n_layers=5
# hidden_size=256
# dropout=0.00041285101332442343

class BPNetModel1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(BPNetModel1, self).__init__()
        self.hiddden = torch.nn.Linear(n_feature, n_hidden)  # ������������
        self.out = torch.nn.Linear(n_hidden, n_output)  # �������������

    def forward(self, x):
        x = Fun.relu(self.hiddden(x))  # ���㼤�������relu()����
        # x=self.hiddden(x)
        # out = Fun.softmax(self.out(x), dim=1)  # ��������softmax����
        out = self.out(x)
        return out

class BPNetModel2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(BPNetModel2, self).__init__()
        self.hiddden1 = torch.nn.Linear(n_feature, n_hidden1)  # ������������
        self.hiddden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # ������������
        self.out = torch.nn.Linear(n_hidden2, n_output)  # �������������

    def forward(self, x):
        x = Fun.relu(self.hiddden1(x))  # ���㼤�������relu()����
        x = Fun.relu(self.hiddden2(x))
        # out = Fun.softmax(self.out(x), dim=1)  # ��������softmax����
        out = self.out(x)
        return out

class BPNetModel3(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2,n_hidden3, n_output):
        super(BPNetModel3, self).__init__()
        self.hiddden1 = torch.nn.Linear(n_feature, n_hidden1)  # ������������
        self.hiddden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # ������������
        self.hiddden3 = torch.nn.Linear(n_hidden2, n_hidden3)  # ������������
        self.out = torch.nn.Linear(n_hidden3, n_output)  # �������������

    def forward(self, x):
        x = Fun.relu(self.hiddden1(x))  # ���㼤�������relu()����
        x = Fun.relu(self.hiddden2(x))
        x = Fun.relu(self.hiddden3(x))
        # out = Fun.softmax(self.out(x), dim=1)  # ��������softmax����
        out = self.out(x)
        return out

class BPNetModel4(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2,n_hidden3,n_hidden4, n_output):
        super(BPNetModel4, self).__init__()
        self.hiddden1 = torch.nn.Linear(n_feature, n_hidden1)  # ������������
        self.hiddden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # ������������
        self.hiddden3 = torch.nn.Linear(n_hidden2, n_hidden3)  # ������������
        self.hiddden4 = torch.nn.Linear(n_hidden3, n_hidden4)  # ������������
        # self.res = torch.nn.Linear(n_feature, n_output)
        self.out = torch.nn.Linear(n_hidden4, n_output)  # �������������

    def forward(self, x):
        # x_res = self.res(x)
        x = Fun.relu(self.hiddden1(x))  # ���㼤�������relu()����
        x = Fun.relu(self.hiddden2(x))
        x = Fun.relu(self.hiddden3(x))
        x = Fun.relu(self.hiddden4(x))
        # out = Fun.softmax(self.out(x), dim=1)  # ��������softmax����
        out = self.out(x)
        # out = Fun.relu(self.out(x))
        return out

curr_maxR2=0
def train(x_train,y_train,x_test,y_test):
    global curr_maxR2
    # net = BPNetModel1(n_feature=n_feature, n_hidden=n_hidden, n_output=n_output)  # ��������
    # net = BPNetModel2(n_feature=n_feature, n_hidden1=n_hidden1,n_hidden2=n_hidden2, n_output=n_output)  # ��������
    # net = BPNetModel3(n_feature=n_feature, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_hidden3=n_hidden3,n_output=n_output)  # ��������
    # net = BPNetModel4(n_feature=n_feature, n_hidden1=n_hidden1,n_hidden2=n_hidden2,n_hidden3=n_hidden3,n_hidden4=n_hidden4, n_output=n_output)  # ��������
    if n_layers==1:
        net = BPNetModel1(n_feature=n_feature, n_hidden1=n_hidden1, n_output=n_output)  # ��������
    if n_layers == 2:
        net = BPNetModel2(n_feature=n_feature, n_hidden1=n_hidden1,n_hidden2=n_hidden2, n_output=n_output)  # ��������
    if n_layers == 3:
        net = BPNetModel3(n_feature=n_feature, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_hidden3=n_hidden3,n_output=n_output)  # ��������
    if n_layers == 4:
        net = BPNetModel4(n_feature=n_feature, n_hidden1=n_hidden1,n_hidden2=n_hidden2,n_hidden3=n_hidden3,n_hidden4=n_hidden4, n_output=n_output)  # ��������

    optimizer = torch.optim.AdamW(net.parameters(), lr= lr,weight_decay=1e-2)  # ʹ��Adam�Ż�����������ѧϰ��
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=step_size)
    loss_fun = torch.nn.MSELoss()  #
    L1_fun = torch.nn.L1Loss()
    loss_steps = []  # ����һ��array([ 0., 0., 0., 0., 0.])������epochs��0
    accuracy_steps = []
    R2=[]
    R2_last_n_list = []
    max_R2 = float('-inf')
    optimizer.zero_grad()
    for epoch in range(epochs):
        y_pred = net(x_train)  # ǰ�򴫲�
        loss = loss_fun(y_pred.reshape(-1), y_train)  # Ԥ��ֵ����ʵֵ�Ա�
        optimizer.zero_grad()  # �ݶ�����
        loss.backward()  # ���򴫲�
        optimizer.step()  # �����ݶ�
        lr_scheduler.step()
        loss_steps.append(loss.item())  # ����loss
        running_loss = loss.item()
        if epoch%100==0:
            print(f"��{epoch}��ѵ����loss={running_loss}".format(epoch, running_loss))
        with torch.no_grad():  # ������û���ݶȵļ���,��Ҫ�ǲ��Լ�ʹ�ã�����Ҫ�ټ����ݶ���
            y_pred = net(x_test)
            L1loss = torch.abs(y_pred-y_test.reshape(-1, 1))
            accuracy_steps.append(L1loss.mean())

            R2.append(r2_score(y_test.reshape(-1).detach().numpy(),y_pred.reshape(-1).detach().numpy()))
        max_R2 = max(max_R2, R2[-1])
        R2_last_n_list.append(R2[-1])
        if len(R2_last_n_list) > 200:
            R2_last_n_list.pop(0)
        max_R2_over_last_n_epochs = max(R2_last_n_list)
        if curr_maxR2 < max_R2:
            curr_maxR2 = max_R2
            torch.save({'model': net.state_dict()}, 'model.pth')
        if max_R2 > max_R2_over_last_n_epochs:
            break
    print('L1loss_std:',np.std(L1loss.detach().numpy()))
    print(torch.sum(L1loss<0.2)/len(y_test))
    print(min(accuracy_steps))

    print('R2:',R2[-1])
    return accuracy_steps[-1],max_R2,(torch.sum(L1loss<0.2)/len(y_test)).detach().numpy(),(torch.sum(L1loss<0.1)/len(y_test)).detach().numpy()

def disrupt(x):
    x_=torch.zeros_like(x)
    rind = [i for i in range(len(x))]
    random.shuffle(rind)
    for i in range(len(x)):
        x_[i].copy_(x[rind[i]])
    return x_

L1_list=[]
R2_list=[]
less_than02_list=[]
less_than01_list=[]
R2_disrupt_list=[]
descent_list=[]
cut_num=int(len(x)/10)
for i in range(10):
    print('������֤:',i+1)
    x_test=x.index_select(0,torch.tensor(rind[i*cut_num:(i+1)*cut_num]))
    y_test=y.index_select(0,torch.tensor(rind[i*cut_num:(i+1)*cut_num]))
    x_train=x.index_select(0,torch.tensor(rind[:i*cut_num]+rind[(i+1)*cut_num:]))
    y_train=y.index_select(0,torch.tensor(rind[:i*cut_num]+rind[(i+1)*cut_num:]))
    L1,R2,less_than02,less_than01=train(x_train, y_train, x_test, y_test)

    # x_train[:,300]=disrupt(x_train[:,300])
    # x_train[18] = disrupt(x_train[18])
    # _, R2_disrupt, _, _ = train(x_train, y_train, x_test, y_test)
    #�Աȷ���
    # regressor = LinearRegression()
    # poly_reg = PolynomialFeatures(degree=2)
    # x_train = poly_reg.fit_transform(x_train)
    # regressor = LinearRegression()

    # regressor = SVR()
    # regressor = DecisionTreeRegressor()#random_state=20
    # regressor = LassoCV()
    # regressor = RidgeCV()

    # regressor = RandomForestRegressor()
    # regressor = LogisticRegression()

    # regressor.fit(x_train, y_train)
    # L1=mean_absolute_error(y_test,regressor.predict(x_test))
    # R2=r2_score(y_test,regressor.predict(x_test))

    L1_list.append(L1)
    R2_list.append(R2)
    # R2_disrupt_list.append(R2_disrupt)
    # descent_list.append(R2-R2_disrupt)
print(x.shape)
# print('R2',R2_list)
print('mean_L1loss:',np.mean(L1_list))
print('mean_R2:',np.mean(R2_list))
print('std_R2:',np.std(R2_list))
time_end=time.time()
print('��������ʱ��:',time_end-time_start)