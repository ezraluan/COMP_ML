import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载数据
data = pd.read_csv('TrainingDataMulti.csv')

# 分离特征和目标
X = data.iloc[:, :128]
y = data.iloc[:, 128]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化分类器
clf = QuadraticDiscriminantAnalysis()

# 训练分类器
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 计算并打印各项指标
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred, average="macro")}')
print(f'Precision: {precision_score(y_test, y_pred, average="macro")}')
print(f'Recall: {recall_score(y_test, y_pred, average="macro")}')
print(f'Error Rate: {1 - accuracy_score(y_test, y_pred)}')

# 计算混淆矩阵并绘制
cm = confusion_matrix(y_test, y_pred)

# 使用热力图显示
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
