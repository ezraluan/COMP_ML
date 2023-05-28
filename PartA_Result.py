from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd

# 1. Load Training Data
df_train = pd.read_csv('TrainingDataBinary.csv', header=None)
X_train = df_train.iloc[:, :128]    # Take the first 128 columns of features
y_train = df_train.iloc[:, 128]     # Get 129 columns of label entries

# 2. Load Testing Data
df_test = pd.read_csv('TestingDataBinary.csv', header=None)
X_test = df_test.iloc[:, :128]  # Assume TestingDataBinary.csv only has feature columns

# Ensure the feature names match
X_test.columns = X_train.columns

# 3. Create random forest classifier model and train
clf = RandomForestClassifier(n_estimators=46, random_state=42, max_depth=30, min_samples_split=2)
clf.fit(X_train, y_train)

# 4. Prediction with test set
y_pred = clf.predict(X_test)

# 5. Add the predicted results to the test set DataFrame and save as a new csv file
X_test['Predicted'] = y_pred
X_test.to_csv('TestingResultsBinary.csv', index=False)
