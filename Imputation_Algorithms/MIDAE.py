import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import MinMaxScaler


def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    # 读取数据
    df_copy = pd.read_csv(input_file)

    # 处理非数值列
    if nonnumerical_column != "None":
        df = df_copy.drop(columns=[nonnumerical_column])
        nonnumerical_data = df_copy[nonnumerical_column].copy()
    else:
        df = df_copy
        nonnumerical_data = None

    # 获取所有数值列的数据
    X = df.values.astype(np.float32)
    n_samples, n_features = X.shape

    # 创建缺失值掩码 (1表示有值, 0表示缺失)
    missing_mask = 1 - np.isnan(X).astype(np.float32)

    print(f"原始数据形状: {X.shape}")
    print(f"每列缺失值数量: {np.sum(1 - missing_mask, axis=0)}")
    print(f"总缺失值数量: {np.sum(1 - missing_mask)}")

    # ============ 数据预处理 ============
    # 使用MinMaxScaler归一化到[0,1]范围
    scaler = MinMaxScaler()

    # 临时用均值填充NaN以便归一化
    X_temp = X.copy()
    col_means = np.nanmean(X, axis=0)
    for j in range(n_features):
        mask = np.isnan(X[:, j])
        X_temp[mask, j] = col_means[j]

    # 归一化
    X_scaled = scaler.fit_transform(X_temp)

    # 重新应用缺失值掩码
    X_scaled = X_scaled * missing_mask + np.nan * (1 - missing_mask)

    print(f"\n数据归一化完成，范围: [0, 1]")

    # ============ MIDAE参数设置 ============
    # 根据论文要求
    learning_rate = 1e-4  # 学习率 1e-4
    batch_size = 64  # mini-batch大小 64
    epochs = 200  # 迭代次数 200
    encoding_dim = 128  # 编码维度（每层128个单元）
    noise_factor = 0.5  # 噪声因子

    print(f"\nMIDAE参数设置:")
    print(f"学习率: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"迭代次数: {epochs}")
    print(f"编码维度: {encoding_dim} (2层，每层128个单元)")
    print(f"噪声因子: {noise_factor}")

    # ============ MIDAE模型定义 ============
    # MIDAE是一个2层网络，每层128个单元
    input_dim = n_features

    # 编码器部分
    input_layer = layers.Input(shape=(input_dim,))

    # 第一层：128个单元
    encoded_1 = layers.Dense(encoding_dim, activation='relu')(input_layer)
    # 第二层：128个单元
    encoded_2 = layers.Dense(encoding_dim, activation='relu')(encoded_1)

    # 解码器部分
    # 第三层：128个单元
    decoded_1 = layers.Dense(encoding_dim, activation='relu')(encoded_2)
    # 输出层：恢复到原始维度
    decoded_output = layers.Dense(input_dim, activation='sigmoid')(decoded_1)

    # 创建自编码器模型
    autoencoder = models.Model(input_layer, decoded_output)

    # 使用ADAM优化器，学习率1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

    print(f"\nMIDAE模型创建成功:")
    print(f"输入维度: {input_dim}")
    print(f"隐藏层1: {encoding_dim} 单元")
    print(f"隐藏层2: {encoding_dim} 单元")
    print(f"输出维度: {input_dim}")
    autoencoder.summary()

    # ============ 训练数据准备 ============
    # 对于MIDAE，我们需要处理所有样本的缺失值
    # 创建一个完整的矩阵用于训练，缺失值先用均值填充
    X_train = X_scaled.copy()
    for j in range(n_features):
        col_mean = np.nanmean(X_scaled[:, j])
        if np.isnan(col_mean):
            col_mean = 0
        mask = np.isnan(X_scaled[:, j])
        X_train[mask, j] = col_mean

    # 添加噪声
    X_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_noisy = np.clip(X_noisy, 0.0, 1.0)

    print(f"\n开始MIDAE训练，共 {epochs} 轮...")

    # ============ 训练模型 ============
    history = autoencoder.fit(
        X_noisy, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.2,
        verbose=1
    )

    # ============ 多次预测取平均 ============
    c = 10  # 预测次数
    print(f"\n进行 {c} 次预测取平均...")

    imputed_matrices = []
    for i in range(c):
        # 每次预测使用不同的噪声
        X_noisy_pred = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
        X_noisy_pred = np.clip(X_noisy_pred, 0.0, 1.0)

        imputed_matrix = autoencoder.predict(X_noisy_pred, verbose=0)
        imputed_matrices.append(imputed_matrix)

        if (i + 1) % 2 == 0:
            print(f"完成 {i + 1}/{c} 次预测")

    # 计算平均填充矩阵
    final_imputed_matrix = np.mean(imputed_matrices, axis=0)

    # ============ 反归一化 ============
    final_imputed_matrix = scaler.inverse_transform(final_imputed_matrix)

    # ============ 创建填充后的DataFrame ============
    imputed_df = pd.DataFrame(final_imputed_matrix, columns=df.columns)

    # 确保原始有值的位置保持不变
    for j, col_name in enumerate(df.columns):
        observed_mask = ~np.isnan(X[:, j])
        if np.any(observed_mask):
            imputed_df.loc[observed_mask, col_name] = X[observed_mask, j]

    # ============ 重新构建完整DataFrame ============
    if nonnumerical_data is not None:
        # 重新添加非数值列
        result_df = pd.concat([nonnumerical_data.to_frame(), imputed_df], axis=1)
        result_df.columns = [nonnumerical_column] + list(imputed_df.columns)
    else:
        result_df = imputed_df

    # 保存结果
    result_df.to_csv(output_file, index=False)

    # ============ 打印统计信息 ============
    print(f"\n填充完成统计:")
    print(f"原始数据范围: [{np.nanmin(X):.2f}, {np.nanmax(X):.2f}]")
    print(f"填充后数据范围: [{np.min(final_imputed_matrix):.2f}, {np.max(final_imputed_matrix):.2f}]")
    print(f"总缺失值数量: {np.sum(1 - missing_mask)}")
    print(f"总填充值数量: {np.sum(1 - missing_mask)}")

    # 计算每列的填充统计
    print(f"\n每列填充统计:")
    for j, col_name in enumerate(df.columns):
        missing_count = np.sum(1 - missing_mask[:, j])
        if missing_count > 0:
            original_min = np.nanmin(X[:, j])
            original_max = np.nanmax(X[:, j])
            filled_min = np.min(final_imputed_matrix[missing_mask[:, j] == 0, j])
            filled_max = np.max(final_imputed_matrix[missing_mask[:, j] == 0, j])
            print(f"{col_name}: 填充 {missing_count} 个值 "
                  f"(原始范围 [{original_min:.2f}, {original_max:.2f}], "
                  f"填充范围 [{filled_min:.2f}, {filled_max:.2f}])")

    print(f'\n{output_file} has been saved.')


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)