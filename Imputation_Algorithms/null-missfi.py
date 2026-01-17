import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sys

def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    df = pd.read_csv(input_file)
    if nonnumerical_column != "None":
        nonnumerical_data = df[nonnumerical_column].copy()
        df.drop(nonnumerical_column, axis=1, inplace=True)


    mask = df[target_column].isnull()
    df_incomplete = df[mask]
    df_complete = df[~mask]

    X_train_initial = df_complete.drop(target_column, axis=1)
    y_train_initial = df_complete[target_column]
    X_test_initial = df_incomplete.drop(target_column, axis=1)

    rfc_initial = RandomForestRegressor(n_estimators=100, random_state=42)
    rfc_initial.fit(X_train_initial, y_train_initial)
    y_pred_initial = rfc_initial.predict(X_test_initial)
    df.loc[mask, target_column] = y_pred_initial

    previous_imputed_values = df[target_column].copy()
    average_difference = 0
    iteration = 0
    max_iterations = 100

    # Modify because there are only missing values in one attribute, making the iterative process meaningless,
    # so change to iteration between each tuple
    while iteration < max_iterations:
        current_imputed_values = df[target_column].copy()
        for index, row in df_incomplete.iterrows():
            print(f"Processing {index}...")
            df_temp = df.drop(index)
            X_train = df_temp.drop(target_column, axis=1)
            y_train = df_temp[target_column]
            X_test = pd.DataFrame([row.drop(target_column)], columns=X_train.columns)

            rfc = RandomForestRegressor(n_estimators=100, random_state=42)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            df.loc[index, target_column] = y_pred[0]

        current_difference = np.mean(np.abs(df[target_column] - previous_imputed_values))
        print(f"Iteration {iteration + 1}: {current_difference},{average_difference}")

        if iteration > 1 and current_difference > average_difference:
            print("Stopping criterion met: average difference increased.")
            df[target_column] = previous_imputed_values
            break

        previous_imputed_values = current_imputed_values
        average_difference = current_difference
        iteration += 1

    if nonnumerical_column != "None":
        df = pd.concat([nonnumerical_data.to_frame(), df], axis=1)
        df.columns.values[0] = nonnumerical_column
    df.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)