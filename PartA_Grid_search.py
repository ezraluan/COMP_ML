from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. loading data
df = pd.read_csv('TrainingDataBinary.csv')
X = df.iloc[:, :128]
y = df.iloc[:, 128]

# 2. divide the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. define the parameter grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [10, 30, 60],
    'min_samples_split': [2, 10, 20]
}
# 4. create the random forest classifier model
clf = RandomForestClassifier(random_state=42)

# 5. create the grid search object and train it
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 6. output the optimal parameters
print('Best Parameters: ', grid_search.best_params_)
