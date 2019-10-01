from sklearn import svm  # svm支持向量机
import matplotlib.pyplot as plt # 可视化绘图
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.externals import joblib

# 读取数据
data = pd.read_csv('data/data0.csv')  # 49×11
data=np.array(data)
# print('data\n', data)
rate=[]
Kern='poly'


for k in range(3):

    # 划分训练集测试集
    X_train,X_test, y_train, y_test  = train_test_split(data[:, 1:11], data[:, 11], test_size=0.3, random_state=k)
    # print('x_train\n', X_train)
    # print('x_test\n', X_test)
    # print('y_train\n', y_train)
    # print('y_test\n', y_test)

    # 搭建svm模型
    svm_clf = svm.SVC(C=1.0, kernel=Kern, probability=True, tol=0.0001, max_iter=100000, degree=3)   #

    # 训练模型
    print('train beagin...')
    svm_clf.fit(X_train, y_train)
    # joblib.dump(svm_clf, "玉米svm1.0.m")
    print('Training finished')

    # 测试模型
    y_pre = svm_clf.predict(X_test)
    print('y_pre', y_pre)
    print('y_true', y_test)

    # 计算准确率
    sum=0
    for i in range(len(y_test)):
        if y_pre[i]==y_test[i]:
            sum=sum+1
    print('len(y_test)', len(y_test))
    print('right sum', sum)
    print('rate[', k, '] ', sum/len(y_test))
    rate.append(sum/len(y_test))
    print('')

print('所用核为：',Kern)
print('\nrate[0]', rate[0], 'rate[1]', rate[1], 'rate[2]', rate[2], 'mean rate', (rate[0]+rate[1]+rate[2])/3)



