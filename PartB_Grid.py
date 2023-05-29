import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier

# Load data
data = pd.read_csv('TrainingDataMulti.csv')
X = data.iloc[:, :128]
y = data.iloc[:, 128]

# Divide the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.2, 0.3, 0.4],
    'max_depth': [7, 9, 11]
}

# Create classifier
clf = LGBMClassifier()

# Create a grid search object
grid_search = GridSearchCV(clf, param_grid, cv=5)

# Performing a grid search
grid_search.fit(X_train, y_train)

# Print optimal parameters
print(grid_search.best_params_)
