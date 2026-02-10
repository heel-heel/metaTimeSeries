import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def process_and_fill(input_file, output_file, target_column, time_column):
    df = pd.read_csv(input_file)

    time_series = df[time_column]
    columns_to_process = [col for col in df.columns if col != time_column]
    data_to_process = df[columns_to_process]

    # 创建缺失位置掩码
    missing_mask = data_to_process.isna()

    iterative_time = 20
    imputed_data_list = []

    for i in range(iterative_time):
        imputer = IterativeImputer(initial_strategy='mean', random_state=i)
        imputed_data_to_process = imputer.fit_transform(data_to_process)
        imputed_data_list.append(imputed_data_to_process)

    imputed_data_avg = np.mean(imputed_data_list, axis=0)
    imputed_data_avg_df = pd.DataFrame(imputed_data_avg, columns=columns_to_process)

    result_data = data_to_process.copy()
    result_data = result_data.mask(missing_mask, imputed_data_avg_df)
    imputed_df_to_process = pd.DataFrame(result_data, columns=columns_to_process)
    imputed_df = pd.concat([time_series, imputed_df_to_process], axis=1)

    imputed_df.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    time_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, time_column)