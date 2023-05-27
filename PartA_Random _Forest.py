from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd

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

# 5. Output accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)

# 6. Output F1 score
f1 = f1_score(y_test, y_pred)
print('F1 Score: ', f1)

# 7. Output error rate
error_rate = 1 - accuracy
print('Error Rate: ', error_rate)

# 8. Add the real tag and predicted results 
#    to the test set DataFrame and save as a new csv file
X_test['True_Label'] = y_test.values
X_test['Predicted'] = y_pred
X_test.to_csv('res.csv', index=False)
