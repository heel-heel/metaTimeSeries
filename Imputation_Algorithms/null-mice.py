import pandas as pd
import numpy as np
import os
from statsmodels.imputation import mice

def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    df = pd.read_csv(input_file)
    if nonnumerical_column != "None":
        df_impute = df.drop(columns=[nonnumerical_column]).copy()
    else:
        df_impute = df.copy()
    np.random.seed(0)
    imp = mice.MICEData(df_impute)
    n_imputations = 50
    imputed_data = []

    for _ in range(n_imputations):
        imp.update(target_column)
        imputed_data.append(imp.data[target_column].copy())
    df_imputed = pd.concat(imputed_data, axis=1).mean(axis=1)
    df.loc[df[target_column].isna(), target_column] = df_imputed
    df.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')



if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)