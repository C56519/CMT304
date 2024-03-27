import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1 加载数据集
file_path = 'measurements.csv'
data = np.loadtxt(file_path, delimiter=',')
time = np.array(data[:, 0])
voltage = np.array(data[:, 1])

# 2 数据预处理，归一化时间以便更好的训练性能
# 最小-最大归一化
time_normalized = (time - np.min(time)) / (np.max(time) - np.min(time))


# 3 划分训练集和测试集(30%)
# 3.1 定义函数
def split_dataset(inputs, labels, test_ratio=0.3):
    # 确定测试集的大小
    total_size = inputs.shape[0]
    test_size = int(total_size * test_ratio)

    # 打乱数据
    indices = np.arange(total_size)
    np.random.seed(42)
    np.random.shuffle(indices)
    inputs_shuffled = inputs[indices]
    labels_shuffled = labels[indices]

    # 划分
    x_test = inputs_shuffled[:test_size]
    y_test = labels_shuffled[:test_size]
    x_train = inputs_shuffled[test_size:]
    y_train = labels_shuffled[test_size:]
    return (x_train, y_train), (x_test, y_test)


# 3.2 使用函数划分数据集
(x_train, y_train), (x_test, y_test) = split_dataset(time, voltage, 0.3)


# 3.3 转成RNN训练数据
def transformToRnn(time, voltage, n_steps):
    X, y = [], []
    for i in range(n_steps, len(time)):
        X.append(time[i - n_steps: i])  # 获取前 n_steps 个电压值如0-9
        y.append(voltage[i])  # 预测的下一个电压值如10
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # 将X转换为(RNN需要的形状)
    return X, y


n_steps = 100
x_train_rnn, y_train_rnn = transformToRnn(x_train, y_train, n_steps)
x_full_rnn, y_full_rnn = transformToRnn(time, voltage, n_steps)

# 4 部署神经网络

# 4.1 配置早期停止
# patience=10: 如果在连续 10 个 epoch 中验证集的损失没有降低，则停止训练
# restore_best_weights=True，表示在停止训练后选出验证集上性能最佳的模型参数
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
# 4.2 创建keras神经网络顺序模型
model = tf.keras.models.Sequential()
# 4.3 规定各层
# 批量归一化层
"""
#model.add(tf.keras.layers.BatchNormalization())
# 全连接层(Dense层) 300个神经元，激活函数relu，并使用了正则化强度为0.01的L2正则化器
model.add(tf.keras.layers.Dense(5, activation="relu", input_shape=(1,), kernel_regularizer=tf.keras.regularizers.l2(0.001)))
#model.add(tf.keras.layers.Dense(2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
"""

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.LSTM(units=32, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.6))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.LSTM(units=32))
model.add(tf.keras.layers.Dropout(0.6))
model.add(tf.keras.layers.Dense(1))

# 4.4 编译神经网络模型，并配置指标
# 损失函数为均方误差
# 模型函数: 随机梯度下降（SGD）优化器来最小化 loss, 并配置学习率为0.5
# 评估指标为均方误差
model.compile(loss="mse", optimizer="adam")
# 4.5 开始训练
# 进行30轮训练，每次的验证集比例0.2，配置回调函数进行早期停止，verbose=1打印每一轮的进度条和训练结果
history = model.fit(x_train_rnn, y_train_rnn, epochs=50, batch_size=32, validation_split=0.2,
                    callbacks=[early_stopping_cb], verbose=1)

# 5 测试
"""
# 测试集上的表现
test_loss= model.evaluate(x_test, y_test)
# 打印模型在测试集上的表现
print(f"\nModel performance in Test Set:")
print(f"Loss: {test_loss}")
"""
# 全部数据上的表现
test_loss = model.evaluate(x_full_rnn, y_full_rnn)
# 打印模型在全部数据集上的表现
print(f"\nModel performance in full dataset:")
print(f"Loss: {test_loss}")
# 输出所有预测值
predictions = model.predict(x_full_rnn)[:, -1]
print(predictions.shape)
np.savetxt('predictions.csv', predictions)

# 6 绘制训练历史和验证历史
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 6.1 绘制时间与真实电压测量值&预测值的图
time_new = time[n_steps:]
print(time_new.shape)
assert len(time_new) == len(predictions), "预测值与时间数据长度不匹配"
plt.figure(figsize=(14, 6))
plt.scatter(time_new, predictions, label='voltage prediction', color='red', s=2)
plt.scatter(time_new, voltage[n_steps:], label='voltage', alpha=0.6, color='blue', s=2)
plt.title('Predicted Voltage & True Voltage')
plt.xlabel('time')
plt.ylabel('voltage')
plt.legend()
plt.show()
