#通过sklearn自带数据包加载iris数据
from sklearn import datasets
iris = datasets.load_iris()
#样本数据与结果分别赋值到x，y
x = iris.data
y = iris.target
#确认样本与结果的输出维度和数据类型
print(x.shape,type(x))
print(y.shape,type(y))
#模型调用
from sklearn.neighbors import KNeighborsClassifier
#创建实例   
knn = KNeighborsClassifier(n_neighbors=1)
print(knn)
#模型训练
knn.fit(x,y)
#模型预测
knn.predict([[1,2,3,4]])
x_test = [[1,2,3,4],[2,4,3,1]]
knn.predict(x_test)
#设置一个新的k值进行KNN建模
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5.fit(x,y)
print(knn_5)
knn_5.predict(x_test)