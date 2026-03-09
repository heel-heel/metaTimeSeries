import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras import layers
from sklearn.preprocessing import StandardScaler


def process_and_fill(input_file, output_file, target_column, time_column):
    # 读取数据
    df = pd.read_csv(input_file)

    # 处理非数值列
    if time_column != "None":
        nonnumerical_data = df[time_column].copy()
        df = df.drop(columns=[time_column])
    else:
        nonnumerical_data = None

    # 获取所有数值列的数据
    X = df.values.astype(np.float32)
    n_samples, n_features = X.shape
    d = n_features  # 特征维度

    print(f"原始数据形状: {X.shape}")
    print(f"特征维度 d = {d}")

    # 创建缺失值掩码 (1表示有值, 0表示缺失)
    missing_mask = 1 - np.isnan(X).astype(np.float32)
    print(f"每列缺失值数量: {np.sum(1 - missing_mask, axis=0)}")
    print(f"总缺失值数量: {np.sum(1 - missing_mask)}")

    # 数据归一化
    scaler = StandardScaler()

    # 临时用均值填充NaN以便归一化
    X_temp = X.copy()
    col_means = np.nanmean(X, axis=0)
    for j in range(n_features):
        mask = np.isnan(X[:, j])
        X_temp[mask, j] = col_means[j]

    # 归一化
    X_scaled = scaler.fit_transform(X_temp)

    # 重新将缺失值设为NaN（但在训练中我们会用掩码处理）
    X_scaled = X_scaled * missing_mask + np.nan * (1 - missing_mask)

    print(f"\n数据归一化完成")

    # ============ GAIN参数设置 ============
    # 根据论文要求
    learning_rate = 1e-4  # 学习率 1e-4
    batch_size = 64  # mini-batch大小 64
    epochs = 200  # 迭代次数 200
    hint_rate = 0.9  # hint rate
    alpha = 10  # 重构损失权重

    print(f"\nGAIN参数设置:")
    print(f"学习率: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"迭代次数: {epochs}")
    print(f"Hint rate: {hint_rate}")
    print(f"Alpha: {alpha}")

    # ============ GAIN模型定义 ============
    # 根据GAIN论文：生成器和判别器都是全连接网络，有两个隐藏层，每层d个神经元
    def make_generator_model():
        model = tf.keras.Sequential([
            layers.Dense(d, activation='relu', input_shape=(d * 2,)),  # 第一隐藏层: d个神经元
            layers.Dense(d, activation='relu'),  # 第二隐藏层: d个神经元
            layers.Dense(d, activation='sigmoid')  # 输出层: d个神经元
        ])
        return model

    def make_discriminator_model():
        model = tf.keras.Sequential([
            layers.Dense(d, activation='relu', input_shape=(d * 2,)),  # 第一隐藏层: d个神经元
            layers.Dense(d, activation='relu'),  # 第二隐藏层: d个神经元
            layers.Dense(d, activation='sigmoid')  # 输出层: d个神经元
        ])
        return model

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    print(f"\nGAIN模型创建成功:")
    print(f"生成器: 输入维度={d * 2}, 隐藏层={d}->{d}")
    print(f"判别器: 输入维度={d * 2}, 隐藏层={d}->{d}")

    # 优化器 - 使用ADAM，学习率1e-4
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # 损失函数
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
    mse_loss = tf.keras.losses.MeanSquaredError()

    # ============ GAIN训练步骤 ============
    @tf.function
    def train_step(X_batch, M_batch):
        batch_size = tf.shape(X_batch)[0]

        # 生成随机噪声 Z
        Z = tf.random.uniform([batch_size, d], minval=0, maxval=1)

        # 生成hint矩阵
        H = tf.random.uniform([batch_size, d], minval=0, maxval=1)
        H = tf.cast(H < hint_rate, tf.float32)  # hint rate概率为1，否则为0
        H = H * M_batch  # hint只在有值的位置有效

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 生成器输入：原始数据（缺失值用Z填充）+ 掩码
            X_tilde = M_batch * X_batch + (1 - M_batch) * Z
            gen_input = tf.concat([X_tilde, M_batch], axis=1)

            # 生成器输出
            G_sample = generator(gen_input, training=True)

            # 最终填充数据
            X_filled = M_batch * X_batch + (1 - M_batch) * G_sample

            # 判别器输入：填充数据 + hint
            disc_input = tf.concat([X_filled, H], axis=1)
            D_prob = discriminator(disc_input, training=True)

            # 判别器损失：只关注缺失值位置的预测
            D_loss = binary_cross_entropy(M_batch, D_prob * M_batch)

            # 生成器损失：包括对抗损失和重构损失
            G_loss_adv = binary_cross_entropy(M_batch, D_prob * M_batch)

            # 重构损失：只计算有值的位置
            G_loss_recon = mse_loss(X_batch * M_batch, G_sample * M_batch)

            # 总生成器损失
            G_loss = G_loss_adv + alpha * G_loss_recon

        # 计算梯度并更新
        gradients_of_generator = gen_tape.gradient(G_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(D_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return G_loss, D_loss, G_loss_adv, G_loss_recon

    # ============ 训练准备 ============
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((X_scaled, missing_mask))
    dataset = dataset.shuffle(n_samples).batch(batch_size)

    print(f"\n开始GAIN训练，共 {epochs} 轮...")

    # ============ 训练循环 ============
    for epoch in range(epochs):
        epoch_g_loss = []
        epoch_d_loss = []
        epoch_g_adv = []
        epoch_g_recon = []

        for X_batch, M_batch in dataset:
            # 处理NaN值（用0临时填充）
            X_batch_clean = tf.where(tf.math.is_nan(X_batch), tf.zeros_like(X_batch), X_batch)

            G_loss, D_loss, G_adv, G_recon = train_step(X_batch_clean, M_batch)

            epoch_g_loss.append(G_loss)
            epoch_d_loss.append(D_loss)
            epoch_g_adv.append(G_adv)
            epoch_g_recon.append(G_recon)

        if (epoch + 1) % 20 == 0:  # 每20轮打印一次
            avg_g_loss = np.mean(epoch_g_loss)
            avg_d_loss = np.mean(epoch_d_loss)
            avg_g_adv = np.mean(epoch_g_adv)
            avg_g_recon = np.mean(epoch_g_recon)
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"G_loss: {avg_g_loss:.4f} (adv: {avg_g_adv:.4f}, recon: {avg_g_recon:.4f}), "
                  f"D_loss: {avg_d_loss:.4f}")

    # ============ 填充所有缺失值 ============
    print("\n开始填充所有缺失值...")

    # 为每个样本生成填充值
    X_filled = X_scaled.copy()

    for i in range(n_samples):
        if i % 100 == 0:
            print(f"正在处理第 {i}/{n_samples} 个样本...")

        # 获取当前样本
        X_i = X_scaled[i:i + 1, :]
        M_i = missing_mask[i:i + 1, :]

        # 处理NaN
        X_i_clean = np.nan_to_num(X_i, nan=0)

        # 生成随机噪声
        Z = np.random.uniform(0, 1, (1, d)).astype(np.float32)

        # 生成器输入
        X_tilde = M_i * X_i_clean + (1 - M_i) * Z
        gen_input = np.concatenate([X_tilde, M_i], axis=1)

        # 生成填充值
        G_sample = generator(gen_input, training=False).numpy()

        # 填充缺失值
        missing_pos = (1 - M_i).astype(bool)
        X_filled[i, missing_pos.flatten()] = G_sample[missing_pos]

    # 反归一化
    X_imputed = scaler.inverse_transform(X_filled)

    # 确保原始有值的位置保持不变
    for j in range(n_features):
        observed_mask = missing_mask[:, j] == 1
        if np.any(observed_mask):
            X_imputed[observed_mask, j] = X[observed_mask, j]

    # 转换回DataFrame
    imputed_df = pd.DataFrame(X_imputed, columns=df.columns)

    # 重新添加非数值列
    if nonnumerical_data is not None:
        imputed_df.insert(0, time_column, nonnumerical_data.values)
        # 非数值列的前向填充
        imputed_df[time_column] = imputed_df[time_column].fillna(method='ffill')

    # 保存结果
    imputed_df.to_csv(output_file, index=False)

    print(f"\n填充完成统计:")
    print(f"原始数据范围: [{np.nanmin(X):.2f}, {np.nanmax(X):.2f}]")
    print(f"填充后数据范围: [{np.min(X_imputed):.2f}, {np.max(X_imputed):.2f}]")
    print(f"总缺失值数量: {np.sum(1 - missing_mask)}")
    print(f"总填充值数量: {np.sum(1 - missing_mask)}")
    print(f'\n{output_file} has been saved.')


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    time_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, time_column)