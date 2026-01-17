import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor


def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    data = pd.read_csv(input_file)
    model = XGBRegressor()
    imputer = IterativeImputer(estimator=model, max_iter=30, random_state=0)
    if nonnumerical_column != "None":
        features = data.columns.drop(nonnumerical_column)
    else:
        features = data.columns
    data_imputed = imputer.fit_transform(data[features])
    data_imputed = pd.DataFrame(data_imputed, columns=features)
    data[target_column] = data_imputed[target_column]
    data.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')



if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)