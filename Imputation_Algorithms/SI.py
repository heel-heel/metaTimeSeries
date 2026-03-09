import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def process_and_fill(input_file, output_file, target_column, time_column):
    df = pd.read_csv(input_file)

    # 保存原始列顺序和列名
    original_columns = df.columns.tolist()

    # 处理非数值列
    non_numeric_col = None
    if time_column in df.columns:
        df_numeric = df.drop(time_column, axis=1)
        non_numeric_col = df[[time_column]]
    else:
        df_numeric = df.copy()

    # 检查哪些列有缺失值
    missing_columns = df_numeric.columns[df_numeric.isnull().any()].tolist()

    # 分离数值数据
    X = df_numeric.values

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 算法参数
    max_iter = 1000
    epsilon = 1e-5

    # 固定阈值（可以直接修改这个值）
    best_threshold = 0.005  # 固定阈值，可以根据需要调整


    # 创建缺失值掩码
    missing_mask = np.isnan(X_scaled)

    # 使用固定阈值填充所有缺失值
    X_final_imputed = X_scaled.copy()
    X_final_imputed[missing_mask] = 0

    for iteration in range(max_iter):
        # 处理可能出现的NaN
        if np.any(np.isnan(X_final_imputed)):
            X_final_imputed = np.nan_to_num(X_final_imputed)

        U, s, Vt = np.linalg.svd(X_final_imputed, full_matrices=False)
        s_thresh = np.maximum(s - best_threshold, 0)
        X_final_imputed_new = U @ np.diag(s_thresh) @ Vt

        diff = np.mean(np.abs(X_final_imputed_new[missing_mask] -
                              X_final_imputed[missing_mask]))
        X_final_imputed = X_final_imputed_new

        if diff < epsilon:
            print(f"Converged after {iteration + 1} iterations")
            break

    # 逆标准化
    X_final = scaler.inverse_transform(X_final_imputed)

    # 更新原始DataFrame中的所有缺失值
    for i, col in enumerate(df_numeric.columns):
        col_missing_mask = np.isnan(df_numeric[col].values)
        if np.any(col_missing_mask):
            # 获取填充后的值
            filled_values = X_final[col_missing_mask, i]
            df.loc[col_missing_mask, col] = filled_values

    # 恢复原始列顺序
    if non_numeric_col is not None:
        df = df[original_columns]

    # 保存结果
    df.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')
    print(f"Filled missing values in columns: {missing_columns}")


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    time_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, time_column)