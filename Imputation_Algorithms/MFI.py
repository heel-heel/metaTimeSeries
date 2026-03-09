import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler


def process_and_fill(input_file, output_file, target_column, time_column):
    data = pd.read_csv(input_file)

    # 处理非数值列
    if time_column != "None":
        numeric_data = data.drop(columns=[time_column])
        non_numeric_data = data[time_column] if time_column in data.columns else None
    else:
        numeric_data = data
        non_numeric_data = None

    # 获取目标列的索引
    target_index = numeric_data.columns.get_loc(target_column)
    X = numeric_data.values

    # 创建所有列的缺失值掩码
    missing_mask_all = np.isnan(X)
    print(f"Missing values per column: {np.sum(missing_mask_all, axis=0)}")
    print(f"Total missing values: {np.sum(missing_mask_all)}")

    # 创建观察值掩码（1表示观察到的值，0表示缺失值）
    M = (~missing_mask_all).astype(int)

    # ============ 逐特征归一化（MinMaxScaler） ============
    n, d = X.shape
    scalers = []  # 保存每个特征的scaler
    X_normalized = X.copy()

    print("\n开始逐特征归一化（MinMaxScaler，范围[0,1]）...")
    for j in range(d):
        # 获取当前特征的非缺失值掩码
        non_missing_mask = ~missing_mask_all[:, j]

        if np.sum(non_missing_mask) > 1:  # 至少有2个非缺失值
            # 创建并拟合scaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(X[non_missing_mask, j].reshape(-1, 1))
            scalers.append(scaler)

            # 归一化非缺失值
            X_normalized[non_missing_mask, j] = scaler.transform(
                X[non_missing_mask, j].reshape(-1, 1)
            ).flatten()

            print(f"列 {numeric_data.columns[j]}: "
                  f"最小值={scaler.data_min_[0]:.4f}, "
                  f"最大值={scaler.data_max_[0]:.4f}, "
                  f"范围={scaler.data_range_[0]:.4f}")
        else:
            # 如果特征没有足够数据，创建虚拟scaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            # 手动设置必要的属性
            scaler.data_min_ = np.array([0])
            scaler.data_max_ = np.array([1])
            scaler.data_range_ = np.array([1])
            scalers.append(scaler)
            print(f"列 {numeric_data.columns[j]}: 数据不足，使用默认归一化")

    # 缺失值位置保持为NaN
    X = X_normalized
    print(f"\n逐特征归一化完成，所有特征范围在[0,1]")

    # 初始化参数
    n, d = X.shape
    p = 2

    # 使用较小的随机值初始化，确保初始预测在合理范围
    U = np.random.randn(n, p) * 0.1  # 正态分布，均值0，标准差0.1
    V = np.random.randn(d, p) * 0.1

    # 添加正则化项，防止过拟合
    lambda_reg = 0.01

    cmax = 100

    # 定义优化目标函数（添加正则化）
    def objective(params):
        U = params[:n * p].reshape(n, p)
        V = params[n * p:].reshape(d, p)
        X_pred = U @ V.T

        # 只计算观察到的值的误差
        observed_mask = M == 1
        if np.sum(observed_mask) > 0:
            observed_diff = np.sum((X[observed_mask] - X_pred[observed_mask]) ** 2)
        else:
            observed_diff = 0

        # 添加L2正则化
        reg_term = lambda_reg * (np.sum(U ** 2) + np.sum(V ** 2))

        return observed_diff + reg_term

    # 定义梯度函数（添加正则化梯度）
    def gradient(params):
        U = params[:n * p].reshape(n, p)
        V = params[n * p:].reshape(d, p)
        X_pred = U @ V.T

        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)

        # 计算梯度
        for i in range(n):
            for j in range(d):
                if M[i, j] == 1:  # 只对观察到的值计算梯度
                    error = X[i, j] - X_pred[i, j]
                    grad_U[i, :] += -2 * error * V[j, :]
                    grad_V[j, :] += -2 * error * U[i, :]

        # 添加正则化梯度
        grad_U += 2 * lambda_reg * U
        grad_V += 2 * lambda_reg * V

        return np.concatenate([grad_U.flatten(), grad_V.flatten()])

    # 定义优化过程
    def optimize(U, V):
        params = np.concatenate([U.flatten(), V.flatten()])

        # 使用L-BFGS-B方法，并启用梯度
        result = minimize(objective, params, method='L-BFGS-B',
                          jac=gradient,
                          options={'maxiter': cmax, 'ftol': 1e-6})

        U_opt = result.x[:n * p].reshape(n, p)
        V_opt = result.x[n * p:].reshape(d, p)
        return U_opt, V_opt

    previous_diffs = []

    for i in range(cmax):
        print("--------------------")
        print(f"Iteration: {i + 1}/{cmax}")

        U, V = optimize(U, V)
        X_pred = U @ V.T

        # 计算观察值上的平均误差
        observed_mask = M == 1
        if np.sum(observed_mask) > 0:
            avg_diff = np.mean((X[observed_mask] - X_pred[observed_mask]) ** 2)
            print(f"Average MSE on observed values (normalized scale [0,1]): {avg_diff:.6f}")

            # 监控预测值范围
            pred_min = np.min(X_pred[observed_mask])
            pred_max = np.max(X_pred[observed_mask])
            print(f"预测值范围: [{pred_min:.4f}, {pred_max:.4f}]")
        else:
            avg_diff = 0
            print("No observed values to compute error")

    # 填充所有缺失值（在归一化后的尺度上）
    X_imputed_normalized = X.copy()
    X_pred = U @ V.T

    # 对每一列进行填充
    for col in range(d):
        col_missing_mask = missing_mask_all[:, col]
        if np.any(col_missing_mask):
            # 关键步骤：将预测值严格限制在[0,1]范围内
            fill_values = X_pred[col_missing_mask, col]
            fill_values = np.clip(fill_values, 0, 1)  # 强制限制在[0,1]
            X_imputed_normalized[col_missing_mask, col] = fill_values
            print(f"Column {numeric_data.columns[col]}: filled {np.sum(col_missing_mask)} missing values "
                  f"(范围: [{np.min(fill_values):.4f}, {np.max(fill_values):.4f}])")

    # ============ 逐特征反归一化 ============
    X_imputed = np.zeros_like(X)

    print("\n开始逐特征反归一化...")
    for j in range(d):
        # 使用对应的scaler进行反向变换
        X_imputed[:, j] = scalers[j].inverse_transform(
            X_imputed_normalized[:, j].reshape(-1, 1)
        ).flatten()

        # 检查反归一化后的范围
        col_min = np.min(X_imputed[:, j])
        col_max = np.max(X_imputed[:, j])
        print(f"列 {numeric_data.columns[j]}: 反归一化范围 [{col_min:.2f}, {col_max:.2f}]")

    # 确保原始观察值保持不变
    for i in range(n):
        for j in range(d):
            if not missing_mask_all[i, j]:  # 如果是原始观察值
                X_imputed[i, j] = numeric_data.values[i, j]  # 恢复原始值

    # 额外的安全检查：确保所有值都在合理范围内
    for j in range(d):
        # 获取原始列的范围
        original_min = scalers[j].data_min_[0]
        original_max = scalers[j].data_max_[0]

        # 允许轻微超出（1%的余量），但严格限制
        lower_bound = original_min - 0.01 * (original_max - original_min)
        upper_bound = original_max + 0.01 * (original_max - original_min)

        # 裁剪超出范围的值
        X_imputed[:, j] = np.clip(X_imputed[:, j], lower_bound, upper_bound)

        # 确保原始观察值不变
        non_missing_mask = ~missing_mask_all[:, j]
        X_imputed[non_missing_mask, j] = numeric_data.values[non_missing_mask, j]

    # 更新数据框
    if non_numeric_data is not None:
        # 如果有非数值列，需要保持原有结构
        for col_idx, col_name in enumerate(numeric_data.columns):
            data.loc[:, col_name] = X_imputed[:, col_idx]
    else:
        # 如果没有非数值列，直接更新所有数值列
        for col_idx, col_name in enumerate(numeric_data.columns):
            data[col_name] = X_imputed[:, col_idx]

    # 保存结果
    data.to_csv(output_file, index=False)
    print(f'\n{output_file} has been saved.')
    print(f"Total missing values filled: {np.sum(missing_mask_all)}")

    # 打印统计信息
    print("\n填充统计信息:")
    print(f"原始数据范围: [{np.nanmin(numeric_data.values):.2f}, {np.nanmax(numeric_data.values):.2f}]")
    print(f"填充后数据范围: [{np.min(X_imputed):.2f}, {np.max(X_imputed):.2f}]")

    # 打印每个特征的归一化参数
    print("\n各特征的MinMax归一化参数:")
    for j, col_name in enumerate(numeric_data.columns):
        print(f"{col_name}: 最小值={scalers[j].data_min_[0]:.4f}, "
              f"最大值={scalers[j].data_max_[0]:.4f}")


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    time_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, time_column)