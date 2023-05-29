import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('TrainingDataMulti.csv')
X = data.iloc[:, :128]
y = data.iloc[:, 128]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameters
n_estimators_range = list(range(150, 251))
learning_rates = [0.2, 0.3, 0.4, 0.5]
max_depth = 7

# Store results
results = {}

# Train the model and record the results
for lr in learning_rates:
    accuracies = []
    for n_est in n_estimators_range:
        clf = LGBMClassifier(n_estimators=n_est, learning_rate=lr, max_depth=max_depth)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    results[lr] = accuracies

# Plot the image
for lr, accuracies in results.items():
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, accuracies, label=f'Learning Rate={lr}')
    plt.title('Model Performance')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
