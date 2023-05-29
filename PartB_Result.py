import pandas as pd
from lightgbm import LGBMClassifier

# Load training data
data = pd.read_csv('TrainingDataMulti.csv', header=None)
X_train = data.iloc[:, :128]
y_train = data.iloc[:, 128]

# Create and train the model
clf = LGBMClassifier(learning_rate=0.3, max_depth=7, n_estimators=150)
clf.fit(X_train, y_train)

# Load test data
test_data = pd.read_csv('TestingResultsMulti.csv', header=None)
X_test = test_data.iloc[:, :128]

# Make predictions
y_pred = clf.predict(X_test)

# Add predictions to the test data
test_data[128] = y_pred

# Save the new CSV file
test_data.to_csv('PartB_res.csv', index=False)
