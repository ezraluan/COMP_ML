from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, log_loss, confusion_matrix
import pandas as pd
import numpy as np

# 加载数据
data = pd.read_csv('TrainingDataMulti.csv')
X = data.iloc[:, 0:128].values
y = data.iloc[:, 128].values

# 数据预处理
encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 定义模型
model = Sequential()
model.add(Dense(64, input_dim=128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=20)

# 评估模型
y_pred = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
print("AUC-ROC: ", roc_auc_score(y_test, y_pred))
print("Log loss: ", log_loss(y_test, y_pred))
print("Confusion matrix: \n", confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
