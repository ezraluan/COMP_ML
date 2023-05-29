import pandas as pd
from lightgbm import LGBMClassifier

# 加载训练数据
data = pd.read_csv('TrainingDataMulti.csv', header=None)
X_train = data.iloc[:, :128]
y_train = data.iloc[:, 128]

# 创建并训练模型
clf = LGBMClassifier(learning_rate=0.3, max_depth=7, n_estimators=150)
clf.fit(X_train, y_train)

# 加载测试数据
test_data = pd.read_csv('TestingResultsMulti.csv', header=None)
X_test = test_data.iloc[:, :128]

# 进行预测
y_pred = clf.predict(X_test)

# 将预测结果添加到测试数据中
test_data[128] = y_pred

# 保存新的CSV文件
test_data.to_csv('PartB_res.csv', index=False)
