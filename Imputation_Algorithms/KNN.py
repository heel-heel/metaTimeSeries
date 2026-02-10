import pandas as pd
import os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def process_and_fill(input_file, output_file, target_column, time_column, k_neighbors=5):
    df = pd.read_csv(input_file)

    # 2. 分离时间列
    if time_column != "None":
        time_series = df[time_column].copy()  # 保存时间列
        data_to_impute = df.drop(columns=[time_column]).copy()
        original_columns = df.drop(columns=[time_column]).columns.tolist()
    else:
        time_series = None
        data_to_impute = df.copy()
        original_columns = df.columns.tolist()

    # 3. 转换为numpy数组
    X = data_to_impute.values.astype(float)  # 确保是浮点型
    n_samples, n_features = X.shape

    # 4. 创建缺失值掩码
    missing_mask = np.isnan(X)

    # 5. ============ 数据归一化 ============

    # 由于存在缺失值，需要为每个特征单独处理
    X_scaled = X.copy()
    scalers = []  # 保存每个特征的scaler

    for feature_idx in range(n_features):
        # 获取当前特征的非缺失值
        non_missing_mask = ~missing_mask[:, feature_idx]

        if np.sum(non_missing_mask) > 1:  # 至少有2个非缺失值
            # 创建并拟合scaler
            scaler = StandardScaler()
            scaler.fit(X[non_missing_mask, feature_idx].reshape(-1, 1))
            scalers.append(scaler)

            # 标准化所有样本（缺失值位置会保持为NaN）
            if np.sum(non_missing_mask) > 0:
                X_scaled[non_missing_mask, feature_idx] = scaler.transform(
                    X[non_missing_mask, feature_idx].reshape(-1, 1)
                ).flatten()
        else:
            # 如果特征没有足够数据，创建虚拟scaler
            scaler = StandardScaler()
            scaler.mean_ = np.array([0])
            scaler.scale_ = np.array([1])
            scalers.append(scaler)

    # 6. 使用零填充的归一化数据
    X_zero_filled = X_scaled.copy()
    X_zero_filled[missing_mask] = 0

    # 7. 按照样本顺序处理
    for sample_idx in range(n_samples):
        # 获取当前样本的缺失特征索引
        missing_features = np.where(missing_mask[sample_idx, :])[0]

        if len(missing_features) == 0:
            continue  # 当前样本没有缺失值

        # 对当前样本的每个缺失特征进行插补
        for feature_idx in missing_features:
            # 7.1 获取当前样本（用于距离计算）
            current_sample = X_zero_filled[sample_idx, :].copy()

            # 创建排除当前特征的索引
            all_indices = np.arange(n_features)
            indices_without_current = np.delete(all_indices, feature_idx)

            # 7.2 找到在当前特征上有观测值的其他样本
            # (排除当前样本自身)
            valid_mask = ~missing_mask[:, feature_idx]
            valid_mask[sample_idx] = False  # 排除自身

            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                # 如果没有有效样本，使用特征均值
                non_missing_values = X_scaled[~missing_mask[:, feature_idx], feature_idx]
                imputed_value = np.mean(non_missing_values) if len(non_missing_values) > 0 else 0
                X_scaled[sample_idx, feature_idx] = imputed_value
                continue

            # 7.3 计算当前样本与所有有效样本的距离（欧几里得距离）
            # 关键修改：排除当前特征列，只使用其他特征计算距离
            distances = cdist(
                current_sample[indices_without_current].reshape(1, -1),
                X_zero_filled[valid_indices[:, None], indices_without_current],
                metric='euclidean'
            ).flatten()

            # 7.4 选择K个最近邻
            actual_k = min(k_neighbors, len(valid_indices))

            # 获取最近邻的索引（在valid_indices中的位置）
            nearest_indices_in_valid = np.argsort(distances)[:actual_k]

            # 转换为原始样本索引
            nearest_indices = valid_indices[nearest_indices_in_valid]

            # 7.5 获取邻居在该特征上的观测值（归一化后的值）
            neighbor_values = X_scaled[nearest_indices, feature_idx]

            # 7.6 计算均值进行插补（仍然在归一化空间）
            imputed_value = np.mean(neighbor_values)

            # 7.7 更新归一化数据
            X_scaled[sample_idx, feature_idx] = imputed_value

    # 8. ============ 反向归一化 ============
    # 使用保存的scaler反向变换
    X_imputed = X.copy()  # 从原始数据开始

    for feature_idx in range(n_features):
        # 使用scaler进行反向变换
        X_imputed[:, feature_idx] = scalers[feature_idx].inverse_transform(
            X_scaled[:, feature_idx].reshape(-1, 1)
        ).flatten()

        # 保持原始非缺失值不变（确保没有数值误差）
        non_missing_mask = ~missing_mask[:, feature_idx]
        X_imputed[non_missing_mask, feature_idx] = X[non_missing_mask, feature_idx]

    # 9. 重构DataFrame
    imputed_df = pd.DataFrame(X_imputed, columns=original_columns)

    # 10. 如果有时间列，重新添加
    if time_series is not None:
        imputed_df.insert(0, time_column, time_series.values)

    # 12. 保存结果
    imputed_df.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    time_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, time_column)