import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, log_loss, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据集
data = pd.read_csv('TrainingDataMulti.csv')

# 前128列是特征，第129列是标签
X = data.iloc[:, 0:128].values
y = data.iloc[:, 128].values

# 将80%的数据作为训练集，20%的数据作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建KNN分类器
knn = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 1, p = 1, weights= 'uniform')

# 训练模型
knn.fit(X_train, y_train)

# 在测试集上预测
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)

# 输出各种性能指标
print(classification_report(y_test, y_pred))

print("AUC-ROC: ", roc_auc_score(label_binarize(y_test, classes=[0, 1, 2]),
                                  y_pred_proba, multi_class='ovo'))
print("Log loss: ", log_loss(label_binarize(y_test, classes=[0, 1, 2]), y_pred_proba))

# 输出混淆矩阵
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))

# 绘制ROC曲线
for i in range(3):
    fpr, tpr, _ = roc_curve(label_binarize(y_test, classes=[0, 1, 2])[:, i], y_pred_proba[:, i])
    plt.plot(fpr, tpr, label='class {}'.format(i))
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# 绘制Precision-Recall曲线
for i in range(3):
    precision, recall, _ = precision_recall_curve(label_binarize(y_test, classes=[0, 1, 2])[:, i], y_pred_proba[:, i])
    plt.plot(recall, precision, label='class {}'.format(i))
plt.title('Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# 手动绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
