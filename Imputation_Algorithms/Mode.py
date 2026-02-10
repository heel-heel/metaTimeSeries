import pandas as pd
import os


def process_and_fill(input_file, output_file, target_column, time_column):
    df = pd.read_csv(input_file)
    columns_to_process = [col for col in df.columns if col != time_column]
    for column in columns_to_process:
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            mode_values = df[column].mode()
            if not mode_values.empty:
                fill_value = mode_values.iloc[0]
            else:
                fill_value = df[column].dropna().iloc[0]
            df[column] = df[column].fillna(fill_value)

    df.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    time_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, time_column)