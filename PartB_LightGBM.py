import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
data = pd.read_csv('TrainingDataMulti.csv')

# Separate features and targets
X = data.iloc[:, :128]
y = data.iloc[:, 128]

# Divide training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Search for parameters
params = {'learning_rate': 0.3, 'max_depth': 7, 'n_estimators': 150}

# Initialize the classifier with the searched parameters
clf = LGBMClassifier(**params, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Perform predictions on the test set
y_pred = clf.predict(X_test)

# Calculate and print metrics
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred, average="macro")}')
print(f'Precision: {precision_score(y_test, y_pred, average="macro")}')
print(f'Recall: {recall_score(y_test, y_pred, average="macro")}')
print(f'Error Rate: {1 - accuracy_score(y_test, y_pred)}')

# Calculate confusion matrix and plot
cm = confusion_matrix(y_test, y_pred)

# Use heat map to display
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
