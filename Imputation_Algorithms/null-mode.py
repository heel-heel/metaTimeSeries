import pandas as pd
import os

def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    df = pd.read_csv(input_file)
    if not df[target_column].empty:
        target_mode = df[target_column].mode()[0]
        df[target_column] = df[target_column].fillna(target_mode)

    df.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)