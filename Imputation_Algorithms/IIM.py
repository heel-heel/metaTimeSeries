import pandas as pd
import numpy as np
import warnings
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def process_and_fill(input_file, output_file, target_column, time_column):
    # 1. 读数据
    df = pd.read_csv(input_file)
    columns_to_process = [col for col in df.columns if col != time_column]

    # 2. 获取缺失掩码矩阵（填补前）
    missing_mask = df[columns_to_process].isna().values
    n_samples, n_features = missing_mask.shape

    # 3. 归一化处理
    X = df[columns_to_process].values.astype(float)
    X_scaled = X.copy()
    scalers = []

    for feature_idx in range(n_features):
        # 获取当前特征的非缺失值
        non_missing_mask = ~missing_mask[:, feature_idx]

        if np.sum(non_missing_mask) > 1:  # 至少有2个非缺失值
            # 创建并拟合scaler
            scaler = StandardScaler()
            scaler.fit(X[non_missing_mask, feature_idx].reshape(-1, 1))
            scalers.append(scaler)

            # 标准化所有样本（缺失值位置保持为NaN）
            if np.sum(non_missing_mask) > 0:
                X_scaled[non_missing_mask, feature_idx] = scaler.transform(
                    X[non_missing_mask, feature_idx].reshape(-1, 1)
                ).flatten()

    # 4. 平均值填补
    for j, column in enumerate(columns_to_process):
        missing_count = missing_mask[:, j].sum()
        if missing_count > 0:
            # 计算原始数据的平均值
            non_missing_values = X[~missing_mask[:, j], j]
            column_mean = np.mean(non_missing_values)

            # 填补原始数据
            X[missing_mask[:, j], j] = column_mean

            # 填补标准化数据
            if scalers[j] is not None:
                # 将平均值转换为标准化值
                scaled_mean = scalers[j].transform([[column_mean]])[0][0]
                X_scaled[missing_mask[:, j], j] = scaled_mean

    # 5. 寻找最优岭回归模型
    optimal_models = []
    neighbor_counts = []
    optimal_models_per_feature = []


    for j in range(n_features):
        feature_models = {}
        # 获取该列原始非缺失的样本索引
        original_non_missing_idx = np.where(~missing_mask[:, j])[0]

        Vj = len(original_non_missing_idx) - 1
        neighbor_counts.append(Vj)

        # 对每个非缺失样本
        for sample_idx in original_non_missing_idx:
            # 当前样本的特征（排除当前列）
            current_features = np.delete(X_scaled[sample_idx], j)

            # 找到该样本的Vj个最近邻（在原始非缺失样本中，排除自身）
            other_samples = [idx for idx in original_non_missing_idx if idx != sample_idx]

            # 计算距离（欧氏距离）
            distances = []
            for neighbor_idx in other_samples:
                neighbor_features = np.delete(X_scaled[neighbor_idx], j)
                dist = np.sqrt(np.sum((current_features - neighbor_features) ** 2))
                distances.append((dist, neighbor_idx))

            # 按距离排序，取前Vj个
            distances.sort(key=lambda x: x[0])
            neighbors = [idx for _, idx in distances[:Vj]]
            print(Vj, len(neighbors))

            # 为每个k（1到邻居数）训练模型并寻找最优模型
            best_model = None
            best_error = float('inf')
            alpha = 0.001
            for k in range(1, len(neighbors) + 1):
                # 使用前k个邻居的特征作为输入（每个邻居的特征单独作为一行）
                X_train_list = []
                y_train_list = []
                for neighbor_idx in neighbors[:k]:
                    neighbor_features = np.delete(X_scaled[neighbor_idx], j)

                    X_train_list.append(neighbor_features)
                    y_train_list.append(X_scaled[sample_idx, j])

                # 转换为数组
                X_train = np.array(X_train_list)
                y_train = np.array(y_train_list)

                model = Ridge(alpha, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict([current_features])[0]
                error = (X_scaled[sample_idx, j] - y_pred) ** 2

                if error < best_error:
                    best_error = error
                    best_model = model
            # 存储这个样本的最优模型
            feature_models[sample_idx] = best_model
        optimal_models_per_feature.append(feature_models)


    # 6. 填补缺失值
    final_imputed = X.copy()
    for j in range(n_features):
        # 找到该列的缺失样本
        missing_indices = np.where(missing_mask[:, j])[0]
        # 找到该列原始非缺失样本作为候选邻居
        candidate_indices = np.where(~missing_mask[:, j])[0]

        for missing_idx in missing_indices:
            current_features = np.delete(X_scaled[missing_idx], j)
            distances = []
            for candidate_idx in candidate_indices:
                candidate_features = np.delete(X_scaled[candidate_idx], j)
                dist = np.sqrt(np.sum((current_features - candidate_features) ** 2))
                distances.append((dist, candidate_idx))

            distances.sort(key=lambda x: x[0])
            top5_neighbors = [idx for _, idx in distances[:min(5, len(distances))]]

            predictions = []

            for neighbor_idx in top5_neighbors:
                if neighbor_idx not in optimal_models_per_feature[j]:
                    print("bad")
                    continue
                neighbor_model = optimal_models_per_feature[j][neighbor_idx]
                # 使用当前缺失样本的特征作为输入
                # 注意：这里用缺失样本的特征，而不是邻居的特征
                missing_input = np.delete(X_scaled[missing_idx], j).reshape(1, -1)
                # 使用邻居的最优模型进行预测
                pred_scaled = neighbor_model.predict(missing_input)[0]

                # 反标准化
                if scalers[j] is not None:
                    pred_original = scalers[j].inverse_transform([[pred_scaled]])[0][0]
                else:
                    pred_original = pred_scaled

                predictions.append(pred_original)

            # 取预测值的平均值
            if len(predictions) > 0:
                final_imputed[missing_idx, j] = np.mean(predictions)

    # 7. 更新数据框
    for j, column in enumerate(columns_to_process):
        missing_idx = np.where(missing_mask[:, j])[0]
        if len(missing_idx) > 0:
            df.loc[missing_idx, column] = final_imputed[missing_idx, j]

    # 8. 保存结果
    df.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')



if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    time_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, time_column)