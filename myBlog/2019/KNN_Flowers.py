# -*- coding: utf-8 -*- 
from sklearn import model_selection 
from sklearn import tree 
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data[:]
y = iris.target
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=1)
k_range = range(1,31)
cv_scores = []  #用来存放每个模型的结果值
num_folds = 10
num_instances = len(X)
seed = 7

for n in k_range:
    knn = KNeighborsClassifier(n)   #knn模型，这里一个超参数可以做预测，当多个超参数时需要使用另一种方法GridSearchCV
    scores = cross_val_score(knn,train_X,train_y,cv=10,scoring='accuracy')  #cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值
    cv_scores.append(scores.mean())
plt.plot(k_range,cv_scores)
plt.xlabel('K')
plt.ylabel('Accuracy')		#通过图像选择最好的参数
#plt.show()
best_knn = KNeighborsClassifier(n_neighbors=3)	# 选择最优的K=3传入模型
best_knn.fit(train_X,train_y)			#训练模型
print(best_knn.score(test_X,test_y))	#看看评分
