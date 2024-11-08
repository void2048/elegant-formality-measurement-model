# elegant-formality-measurement-model
本程序所使用的训练集与测试集数据均是从corpus中生成，train.py和train(BO).py为神经网络训练程序，先使用贝叶斯优化算法（train(BO).py）确认最优超参数的范围，再使用train.py进行进一步调整，优化模型性能。
test.py使用训练好的模型进行庄雅度测量。
