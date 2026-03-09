import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sys


def process_and_fill(input_file, output_file, target_column, time_column):
    # 读取数据
    df = pd.read_csv(input_file)

    # 处理非数值列
    if time_column != "None":
        nonnumerical_data = df[time_column].copy()
        df = df.drop(columns=[time_column])
    else:
        nonnumerical_data = None

    # 转换为numpy数组便于处理
    X_original = df.values
    n_samples, n_features = X_original.shape

    # 创建缺失值掩码 M (1表示观察到的值，0表示缺失值)
    missing_mask = np.isnan(X_original)
    M = (~missing_mask).astype(int)

    print(f"原始数据形状: {X_original.shape}")
    print(f"每列缺失值数量: {np.sum(missing_mask, axis=0)}")
    print(f"总缺失值数量: {np.sum(missing_mask)}")

    # Step 1: 用均值填充初始化 X^0
    X_imputed = X_original.copy()
    col_means = np.nanmean(X_original, axis=0)
    for j in range(n_features):
        mask = missing_mask[:, j]
        X_imputed[mask, j] = col_means[j]

    print("\n初始均值填充完成")

    # 保存列名
    column_names = df.columns.tolist()

    # XGBI参数
    cmax = 100
    threshold = 1000  #Illness # 停止阈值，当平均差异小于此值时停止
    c = 0

    # 记录每列缺失值的索引，避免重复计算
    missing_cols = [j for j in range(n_features) if np.sum(missing_mask[:, j]) > 0]

    print(f"\n需要填充的列: {[column_names[j] for j in missing_cols]}")
    print(f"XGBI停止阈值: {threshold}")

    # 主循环
    while c < cmax:
        c += 1
        print(f"\n==================== 迭代 {c}/{cmax} ====================")

        # X^c = X^{c-1} (复制当前填充结果)
        X_current = X_imputed.copy()

        # 对每个有缺失值的特征进行填充
        for fj in missing_cols:
            feature_name = column_names[fj]
            missing_idx = np.where(missing_mask[:, fj])[0]  # 当前列缺失值的行索引
            observed_idx = np.where(~missing_mask[:, fj])[0]  # 当前列观察值的行索引

            print(f"\n处理特征 '{feature_name}': {len(missing_idx)} 个缺失值")

            if len(observed_idx) < 2:  # 如果观察值太少，跳过
                print(f"  警告: 特征 '{feature_name}' 观察值不足，跳过")
                continue

            # 准备训练数据：使用当前填充的X^c中对应行的所有特征
            X_train = X_current[observed_idx, :]  # 所有特征
            X_train = np.delete(X_train, fj, axis=1)  # 删除当前特征列

            y_train = X_current[observed_idx, fj]  # 当前特征作为目标

            # 准备测试数据：缺失值所在行的所有其他特征
            X_test = X_current[missing_idx, :]
            X_test = np.delete(X_test, fj, axis=1)  # 删除当前特征列

            # 使用随机森林（XGBoost风格但用随机森林实现）
            dt = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                n_jobs=-1,
                random_state=42
            )

            dt.fit(X_train, y_train)

            # 预测缺失值
            y_pred = dt.predict(X_test)

            # 更新X_current中的缺失值
            X_current[missing_idx, fj] = y_pred

            print(f"  填充完成，预测值范围: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")

        # 计算新旧填充值的平均差异（XGBI停止条件）
        diff = np.mean(np.abs(X_current[missing_mask] - X_imputed[missing_mask]))
        print(f"\n迭代 {c} 的平均差异: {diff:.6f}")

        # XGBI停止条件：平均差异小于阈值
        if diff < threshold:
            print(f"\nXGBI停止条件触发: 平均差异({diff:.6f}) < 阈值({threshold})")
            print(f"使用当前迭代的结果 (X^{c})")
            X_imputed = X_current  # 使用当前迭代的结果
            break
        else:
            # 更新X_imputed为当前迭代结果
            X_imputed = X_current.copy()

    if c == cmax:
        print(f"\n达到最大迭代次数 {cmax}")
        print(f"使用最后一次迭代的结果")

    # 将结果转换回DataFrame
    imputed_df = pd.DataFrame(X_imputed, columns=column_names)

    # 重新添加非数值列
    if nonnumerical_data is not None:
        imputed_df.insert(0, time_column, nonnumerical_data.values)

    # 保存结果
    imputed_df.to_csv(output_file, index=False)

    print(f"\n填充完成统计:")
    print(f"总迭代次数: {c}")
    print(f"最终平均差异: {diff:.6f}")
    print(f"停止阈值: {threshold}")
    print(f"原始数据范围: [{np.nanmin(X_original):.2f}, {np.nanmax(X_original):.2f}]")
    print(f"填充后数据范围: [{np.min(X_imputed):.2f}, {np.max(X_imputed):.2f}]")
    print(f"总缺失值数量: {np.sum(missing_mask)}")
    print(f'\n{output_file} has been saved.')


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    time_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, time_column)