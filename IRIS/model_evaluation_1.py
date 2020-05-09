"""
计算k值不同的时候预测结果的准确率
"""
#数据加载
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

#模型训练与预测 k=5
from sklearn.neighbors import KNeighborsClassifier
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5.fit(x,y)
y_pred = knn_5.predict(x)
print(y_pred)
print(y_pred.shape)

#计算准确率
from sklearn.metrics import accuracy_score
print(accuracy_score(y,y_pred))

#模型训练与预测 k=1
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(x,y)
y_pred = knn_1.predict(x)
print(accuracy_score(y,y_pred))
 