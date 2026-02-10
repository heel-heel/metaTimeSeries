import pandas as pd
import os
import random

random.seed(42)

def insert_null(input_file, output_file, rate, exclude_columns=None):
    df = pd.read_csv(input_file)
    if exclude_columns is None:
        exclude_columns = []
    columns_to_insert_null = [col for col in df.columns if col not in exclude_columns]
    for column in columns_to_insert_null:
        num_missing = int(len(df) * rate / 100)
        random_indices = random.sample(range(len(df)), num_missing)
        df.loc[random_indices, column] = pd.NA
    df.to_csv(output_file, index=False)
    print(f'{output_file} has saved.')

datasets = {
    "Illness": {"time_column": "date", "target_column": "OT"},
}
Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
base_path = "../Datasets"

for dataset, config in datasets.items():
    input_file = os.path.join(base_path, dataset, "clean.csv")
    output_path = os.path.join(base_path, dataset, "null_all")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for rate in Missing_rate:
        output_file = os.path.join(output_path, f'null_all-{rate}.csv')
        insert_null(input_file, output_file, rate, exclude_columns=[config['time_column']])