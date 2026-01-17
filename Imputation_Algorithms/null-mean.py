import pandas as pd
import os

def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    data = pd.read_csv(input_file)
    global_mean = data[target_column].mean()
    data[target_column] = data[target_column].fillna(global_mean)
    data.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)