import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. data preparation
data = pd.read_csv('TrainingDataBinary.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 2. splitting the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. defining the resulting DataFrame
df_results = pd.DataFrame()

# 4. define the resultant DataFrame for each max_depth
for max_depth in [None, 10, 20, 30, 40]:
    accuracy_results = []
    # For each n_estimators
    for n_estimators in range(1, 51):
        # Train the model
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        # Predict and calculate accuracy
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results.append(accuracy)
    
    # Save the results to a DataFrame
    df_results[f'max_depth={str(max_depth) if max_depth is not None else "None"}'] = accuracy_results

# 5. Graph the results
for column in df_results.columns:
    plt.plot(range(1, 51), df_results[column], label=column)

plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 6. Save results as CSV file
df_results.to_csv('accuracy_results.csv', index=False)
