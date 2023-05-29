from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
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

# 用模型进行预测，得到softmax层的输出
y_pred = model.predict(X_test)

# 为了得到真实标签，我们需要对one-hot编码的标签进行逆操作
y_test_labels = y_test.argmax(axis=1)

# 将y_pred与对应的标签值结合起来
output_data = np.column_stack((y_pred, y_test_labels))

# 将数据转换为DataFrame，以便于导出为csv文件
output_df = pd.DataFrame(output_data, columns=['Predicted_0', 'Predicted_1', 'Predicted_2', 'True_Label'])

# 输出为csv文件
output_df.to_csv('prediction_output.csv', index=False)
