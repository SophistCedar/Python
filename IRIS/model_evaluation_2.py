"""
把数据集分成两部分：训练集、测试集
使用训练集进行模型训练
使用测试集进行预测，从而评估模型表现
"""
#数据加载
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target
print(x.shape,y.shape)
#数据分离
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)
#分离后数据维度确认
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
#分离后的数据预测与评估 k=5
from sklearn.neighbors import KNeighborsClassifier
knn_5_s = KNeighborsClassifier(n_neighbors=5)
knn_5_s.fit(x_train,y_train)
y_train_pred = knn_5_s.predict(x_train)
y_test_pred = knn_5_s.predict(x_test)
from sklearn.metrics import accuracy_score
print("k=5",accuracy_score(y_train,y_train_pred))
print("k=5",accuracy_score(y_test,y_test_pred))

#分离后的数据预测与评估 k=1
knn_1_s = KNeighborsClassifier(n_neighbors=1)
knn_1_s.fit(x_train,y_train)
y_train_pred = knn_1_s.predict(x_train)
y_test_pred = knn_1_s.predict(x_test)
from sklearn.metrics import accuracy_score
print("k=1",accuracy_score(y_train,y_train_pred))
print("k=1",accuracy_score(y_test,y_test_pred))

