import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score, log_loss
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("TrainingDataMulti.csv")
X = data.iloc[:, :128]
y = data.iloc[:, 128]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 SVM 模型
svm = SVC(kernel='linear', probability=True, random_state=42)

# 训练模型
svm.fit(X_train, y_train)

# 进行预测
y_pred = svm.predict(X_test)

# 输出各项评价指标
print("Classification report:")
print(classification_report(y_test, y_pred))

# 画出混淆矩阵
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
plt.show()

# 为了绘制 ROC 和 PR 曲线，我们需要将标签二值化，并计算各类别的预测概率
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = svm.decision_function(X_test)

# 计算并绘制 ROC 曲线
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure()
for i in range(3):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.show()

# 计算并绘制 PR 曲线
precision = dict()
recall = dict()
for i in range(3):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
plt.figure()
for i in range(3):
    plt.plot(recall[i], precision[i], label='PR curve of class {0}'.format(i))
plt.show()

# 计算并输出其他评价指标
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Log loss: ", log_loss(y_test, svm.predict_proba(X_test)))
