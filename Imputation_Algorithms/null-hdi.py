import pandas as pd
import os
from sklearn.impute import KNNImputer

def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    df=pd.read_csv(input_file)
    imputer = KNNImputer(n_neighbors=1)
    if nonnumerical_column != "None":
        features = df.drop(columns=[nonnumerical_column])
    else:
        features = df
    imputed_data = imputer.fit_transform(pd.concat([df[target_column], features], axis=1))

    df[target_column] = imputed_data[:, 0]
    df.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)