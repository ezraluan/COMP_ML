import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. load data 
df = pd.read_csv('TrainingDataBinary.csv')
X = df.iloc[:, :128]
y = df.iloc[:, 128]

# 2. divide training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. create an empty dataframe to store results
df_results = pd.DataFrame()

# 4. iterate over each min_samples_split value
for min_samples_split in [2, 5, 10]:
    accuracy_results = []

    # 5. iterate over the values of n_estimators
    for n_estimators in range(1, 51):
        # create random forest classifier model and train
        clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=42)
        clf.fit(X_train, y_train)

        # use test set to make predictions and calculate accuracy
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        accuracy_results.append(accuracy)

    df_results['min_samples_split=' + str(min_samples_split)] = accuracy_results

# 6. plot folded line graph
for column in df_results.columns:
    plt.plot(range(1, 51), df_results[column], label=column)

plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.legend()
plt.show()
