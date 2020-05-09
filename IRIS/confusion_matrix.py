"""
皮马印第安人糖尿病数据集
基于数据集中包括的某些诊断测量来诊断性地预测患者是否患有糖尿病
输入变量包括：患者的怀孕次数、葡萄糖量、血压、皮褶厚度、体重指数、胰岛素水平、糖尿病谱系功能、年龄
输出结果：是否患有糖尿病
混淆矩阵指标
准确率 整体样本中，预测正确样本数的比例
Accuracy = (TP+TN)/(TP+TN+FP+FN)
错误率 整体样本中，预测错误样本数的比例
Misclassification Rate = (FP+FN)/(TP+TN+FP+FN)
召回率 正样本中，预测正确的比例
Recall = TP/(TP+FN)
特异度 负样本中，预测正确的比例
Specifity = TN/(TN+FP)
精确率 预测结果为正的样本中，预测正确的比例
Precision = TP/(TP+FP)
F1分数 综合Precision和Recall的一个判断指标
F1 socre = 2*Precision*Recall/(Precision+Recall)
"""
#数据预处理
import pandas as pd
path = 'C:/Users/Administrator/Desktop/iris/diabetes.csv'
pima = pd.read_csv(path)
print(pima.head())
#x和y赋值
feature_names = ['Pregnancies','Insulin','BMI','Age']
x = pima[feature_names]
y = pima.Outcome
#维度确认
print('x.shape',x.shape)
print('y.shape',y.shape)
#数据分离
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
#模型训练
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
#测试数据集结果预测
y_pred = logreg.predict(x_test)
#使用准确率进行评估
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
#确认正负样本数据量
y_test.value_counts()
#1的比例
print('1的比例',y_test.mean())
#0的比例
print('0的比例',1-y_test.mean())
#空准确率
print('空准确率',max(y_test.mean(),1-y_test.mean()))
#计算并展示混淆矩阵
from sklearn import metrics
print('混淆矩阵:')
print(metrics.confusion_matrix(y_test,y_pred))
#展示部分实际结果与预测结果（25组）
print('true',y_test.values[0:25])
print('pred',y_pred[0:25])
#四个因子赋值
confusion = metrics.confusion_matrix(y_test,y_pred)
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
TP = confusion[1,1]
print(TN,FP,FN,TP)

#准确率 整体样本中，预测正确样本数的比例
Accuracy = (TP+TN)/(TP+TN+FP+FN)
print('Accuracy',Accuracy)
print(accuracy_score(y_test,y_pred))
#错误率 整体样本中，预测错误样本数的比例
Misclassification_Rate = (FP+FN)/(TP+TN+FP+FN)
print('Misclassification_Rate',Misclassification_Rate)
print(1-accuracy_score(y_test,y_pred))
#召回率 正样本中，预测正确的比例
Recall = TP/(TP+FN)
print('Recall',Recall)
#特异度 负样本中，预测正确的比例
Specifity = TN/(TN+FP)
print('Specifity',Specifity)
#精确率 预测结果为正的样本中，预测正确的比例
Precision = TP/(TP+FP)
print('Precision',Precision)
#F1分数 综合Precision和Recall的一个判断指标
F1_socre = 2*Precision*Recall/(Precision+Recall)
print('F1_socre',F1_socre)






