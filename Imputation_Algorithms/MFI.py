import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    data = pd.read_csv(input_file)
    if nonnumerical_column != "None":
        numeric_data = data.drop(columns=[nonnumerical_column])
    else:
        numeric_data = data
    target_index = numeric_data.columns.get_loc(target_column)
    X = numeric_data.values

    missing_mask = np.isnan(X[:, target_index])
    M = np.ones_like(X[:, target_index])
    M[missing_mask] = 0

    # Initialize
    p = 5
    n, d = X.shape
    U = np.random.rand(n, p)
    V = np.random.rand(d, p)
    cmax = 100
    #threshold = 10000000  # Convergence threshold, adjustable
    #threshold = 0.0001
    threshold = 0.01# 0.01 for ETT, 0.0001for Exchange, 0.1 for Weather

    # Define the optimization objective function
    def objective(params):
        U = params[:n * p].reshape(n, p)
        V = params[n * p:].reshape(d, p)
        X_pred = U @ V.T
        observed_diff = np.sum((X[M == 1] - X_pred[M == 1]) ** 2)
        return observed_diff

    # Define the optimization process
    def optimize(U, V):
        params = np.concatenate([U.flatten(), V.flatten()])
        result = minimize(objective, params, method='BFGS', options={'maxiter': cmax})
        U_opt = result.x[:n * p].reshape(n, p)
        V_opt = result.x[n * p:].reshape(d, p)
        return U_opt, V_opt

    consecutive_count = 0
    previous_diffs = []

    for i in range(cmax):
        print("--------------------")
        print(f"echo:{i}")
        U, V = optimize(U, V)
        X_pred = U @ V.T
        avg_diff = np.mean((X[M == 1] - X_pred[M == 1]) ** 2)
        print(avg_diff)

        previous_diffs.append(avg_diff)
        print(previous_diffs)
        if len(previous_diffs) > 5:
            previous_diffs.pop(0)

        if len(previous_diffs) == 5:
            diffs = [abs(previous_diffs[j] - previous_diffs[j + 1]) for j in range(len(previous_diffs) - 1)]
            print(diffs)
            if all(diff < threshold for diff in diffs):
                print("Stop iteration when the difference has been less than 1 for 5 consecutive times")
                break

    # imputation
    X_imputed = X.copy()
    X_imputed[missing_mask] = (U @ V.T)[missing_mask]
    if nonnumerical_column != "None":
        data.iloc[:, target_index + 1] = X_imputed[:, target_index]
    else:
        data.iloc[:, target_index] = X_imputed[:, target_index]
    data.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)