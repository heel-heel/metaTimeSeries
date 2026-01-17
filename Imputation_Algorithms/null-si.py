import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    df = pd.read_csv(input_file)

    if target_column in df.columns:
        target_col_index = df.columns.get_loc(target_column)
    else:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    print(f"Target column '{target_column}' is at index: {target_col_index}")

    if nonnumerical_column != "None":
        X = df.drop(nonnumerical_column, axis=1).values
    else:
        X = df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_iter = 1000
    epsilon = 1e-5
    thresholds = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    val_ratio = 0.2
    missing_mask = np.isnan(X_scaled)
    missing_mask_target = missing_mask[:, target_col_index-1]

    rows_with_missing = np.any(missing_mask, axis=1)
    complete_rows = ~rows_with_missing

    if np.sum(complete_rows) < 2:
        raise ValueError("Not enough complete samples for validation. Consider using a different method.")

    X_complete = X_scaled[complete_rows]
    train_idx, val_idx = train_test_split(np.arange(X_complete.shape[0]),
                                          test_size=val_ratio,
                                          random_state=42)

    X_train = X_complete[train_idx]
    X_val = X_complete[val_idx]

    np.random.seed(42)
    val_missing_mask = np.random.rand(*X_val.shape) < 0.2
    X_val_incomplete = X_val.copy()
    X_val_incomplete[val_missing_mask] = np.nan

    X_filled = X_scaled.copy()
    X_filled[missing_mask] = 0

    best_threshold = None
    best_error = np.inf

    for threshold in thresholds:
        X_val_imputed = X_val_incomplete.copy()
        X_val_imputed[np.isnan(X_val_imputed)] = 0

        for _ in range(max_iter):
            U, s, Vt = np.linalg.svd(X_val_imputed, full_matrices=False)
            s_thresh = np.maximum(s - threshold, 0)
            X_val_imputed_new = U @ np.diag(s_thresh) @ Vt

            # 检查收敛
            diff = np.mean(np.abs(X_val_imputed_new[val_missing_mask] -
                                  X_val_imputed[val_missing_mask]))
            X_val_imputed = X_val_imputed_new

            if diff < epsilon:
                break
        val_error = np.nanmean((X_val_imputed[val_missing_mask] -
                                X_val[val_missing_mask]) ** 2)
        print(f"Threshold {threshold}: Validation MSE = {val_error}")
        if val_error < best_error:
            best_error = val_error
            best_threshold = threshold
    print(f"Best threshold selected by cross-validation: {best_threshold}")

    X_final_imputed = X_scaled.copy()
    X_final_imputed[missing_mask] = 0

    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(X_final_imputed, full_matrices=False)
        s_thresh = np.maximum(s - best_threshold, 0)
        X_final_imputed_new = U @ np.diag(s_thresh) @ Vt

        diff = np.mean(np.abs(X_final_imputed_new[missing_mask] -
                              X_final_imputed[missing_mask]))
        X_final_imputed = X_final_imputed_new

        if diff < epsilon:
            break

    X_final = scaler.inverse_transform(X_final_imputed)
    df.loc[missing_mask_target, target_column] = X_final[missing_mask_target, target_col_index-1]
    df.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)