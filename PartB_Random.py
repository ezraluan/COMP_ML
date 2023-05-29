import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from lightgbm import LGBMClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Load data
data = pd.read_csv('TrainingDataMulti.csv')
X = data.iloc[:, :128]
y = data.iloc[:, 128]

# Divide the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter distribution
param_dist = {
    'n_estimators': sp_randint(130, 250),
    'learning_rate': sp_uniform(0.2, 0.5),
    'max_depth': sp_randint(5, 9),
}

# Create classifier
clf = LGBMClassifier()

# Create random search objects
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, cv=5)

# Perform a random search
random_search.fit(X_train, y_train)

# Print the optimal parameters
print(random_search.best_params_)
