import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1 加载数据集
file_path = 'measurements.csv'
data = np.loadtxt(file_path, delimiter=',')
time = data[:, 0]
voltage = data[:, 1]


# 2 数据预处理，归一化时间以便更好的训练性能
# 最小-最大归一化
time_normalized = (time - np.min(time)) / (np.max(time) - np.min(time))


# 3 划分训练集和测试集(30%)
# 3.1 定义函数
def split_dataset(inputs, labels, test_ratio = 0.3):
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
(time_train, voltage_train), (time_test, voltage_test) = split_dataset(time, voltage, 0.3)

# 3.3 转换成tensorflow张量
x_train = tf.convert_to_tensor(time_train[:, np.newaxis], dtype = tf.float32)
y_train = tf.convert_to_tensor(voltage_train, dtype = tf.float32)
x_test = tf.convert_to_tensor(time_test[:, np.newaxis], dtype = tf.float32)
y_test = tf.convert_to_tensor(voltage_test, dtype = tf.float32)

xfull_test = tf.convert_to_tensor(time_normalized[:, np.newaxis], dtype = tf.float32)
yfull_test = tf.convert_to_tensor(voltage, dtype = tf.float32)


# 4 部署神经网络

# 4.1 配置早期停止
# patience=10: 如果在连续 10 个 epoch 中验证集的损失没有降低，则停止训练
# restore_best_weights=True，表示在停止训练后选出验证集上性能最佳的模型参数
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
# 4.2 创建keras神经网络顺序模型
model = tf.keras.models.Sequential()
# 4.3 规定各层
# 批量归一化层
model.add(tf.keras.layers.BatchNormalization())
# 全连接层(Dense层) 300个神经元，激活函数relu，并使用了正则化强度为0.01的L2正则化器
model.add(tf.keras.layers.Dense(300, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(100, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# 4.4 编译神经网络模型，并配置指标
# 损失函数为均方误差
# 模型函数: 随机梯度下降（SGD）优化器来最小化 loss, 并配置学习率为0.5
# 评估指标为均方误差
model.compile(loss = "mse", optimizer="adam")
# 4.5 开始训练
# 进行30轮训练，每次的验证集比例0.2，配置回调函数进行早期停止，verbose=1打印每一轮的进度条和训练结果
history = model.fit(xfull_test, yfull_test, epochs=50, validation_split=0.2, callbacks=[early_stopping_cb], verbose=1)


# 5 测试
"""
# 测试集上的表现
test_loss= model.evaluate(x_test, y_test)
# 打印模型在测试集上的表现
print(f"\nModel performance in Test Set:")
print(f"Loss: {test_loss}")
"""
# 全部数据上的表现
test_loss= model.evaluate(xfull_test, yfull_test)
# 打印模型在全部数据集上的表现
print(f"\nModel performance in full dataset:")
print(f"Loss: {test_loss}")
# 输出所有预测值
predictions = model.predict(xfull_test)
np.savetxt('predictions.csv', predictions)

# 6 绘制训练历史和验证历史
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 6.1 绘制时间与真实电压测量值&预测值的图
plt.figure(figsize=(14, 6))
plt.scatter(time, predictions, label='voltage prediction', color='red', s=2)
plt.scatter(time, voltage, label='voltage', alpha=0.6, color='blue', s=2)
plt.title('Predicted Voltage & True Voltage')
plt.xlabel('time')
plt.ylabel('voltage')
plt.legend()
plt.show()