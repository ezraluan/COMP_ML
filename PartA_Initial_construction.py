from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, log_loss
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Data Loading
df = pd.read_csv('TrainingDataBinary.csv')
X = df.iloc[:, :128]    # Take the first 128 columns of features
y = df.iloc[:, 128]     # Get 129 columns of label entries

# 2. Divide the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create random forest classifier model and train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Prediction with test sets
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]   # get the probability of class 1

# 5. Output accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)

# 6. Output F1 score
f1 = f1_score(y_test, y_pred)
print('F1 Score: ', f1)

# 7. Output error rate
error_rate = 1 - accuracy
print('Error Rate: ', error_rate)

# 8. Precision and Recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Precision: ', precision)
print('Recall: ', recall)

# 9. AUC-ROC
auc_roc = roc_auc_score(y_test, y_prob)
print('AUC-ROC: ', auc_roc)

# 10. Log loss
logloss = log_loss(y_test, y_prob)
print('Log Loss: ', logloss)

# 11. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: ')
print(cm)

# 12. Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_roc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()

# 13. Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(recall, precision, label='PR curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# 14. Plot confusion matrix manually
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# 15. Add the real tag and predicted results 
#     to the test set DataFrame and save as a new csv file
X_test['True_Label'] = y_test.values
X_test['Predicted'] = y_pred
X_test.to_csv('res.csv', index=False)
