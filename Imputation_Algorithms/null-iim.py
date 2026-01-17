import pandas as pd
import numpy as np
import time
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    np.random.seed(42)
    df_copy = pd.read_csv(input_file)
    if nonnumerical_column != "None":
        df = df_copy.drop(columns=nonnumerical_column)
    else:
        df = df_copy

    # complete data and missing data
    complete_df = df.dropna(subset=[target_column])
    incomplete_df = df[df[target_column].isnull()]

    if len(complete_df) == 0:
        raise ValueError("No complete data available for learning")

    feature_columns = [col for col in df.columns if col != target_column]
    scaler = StandardScaler()
    X_complete_scaled = scaler.fit_transform(complete_df[feature_columns])
    X_incomplete_scaled = scaler.transform(incomplete_df[feature_columns])

    # learning stage
    def learning_phase(X_complete, y_complete, l_values):
        models_dict = {}
        n_complete = len(X_complete)

        for i in range(n_complete):
            current_sample = X_complete[i]
            current_value = y_complete[i]
            distances = np.linalg.norm(X_complete - current_sample, axis=1)
            sorted_indices = np.argsort(distances)
            sorted_indices = sorted_indices[sorted_indices != i]

            models_dict[i] = {}

            for l in l_values:
                if l >= len(sorted_indices):
                    continue
                neighbor_indices = sorted_indices[:l]

                if len(neighbor_indices) < 2:
                    if len(neighbor_indices) == 1:
                        models_dict[i][l] = {'type': 'direct', 'value': y_complete[neighbor_indices[0]]}
                    else:
                        models_dict[i][l] = {'type': 'direct', 'value': current_value}
                    continue

                X_neighbors = X_complete[neighbor_indices]
                y_neighbors = y_complete[neighbor_indices]

                try:
                    model = Ridge(alpha=1.0, random_state=42)
                    model.fit(X_neighbors, y_neighbors)

                    # evaluate
                    y_pred = model.predict(X_neighbors)
                    mse = mean_squared_error(y_neighbors, y_pred)

                    models_dict[i][l] = {
                        'type': 'ridge',
                        'model': model,
                        'mse': mse,
                        'neighbor_indices': neighbor_indices
                    }
                except:
                    models_dict[i][l] = {'type': 'mean', 'value': np.mean(y_neighbors)}

        return models_dict

    # choose the best model
    def select_optimal_models(models_dict, X_complete, y_complete, k_validation=5):
        optimal_models = {}
        n_complete = len(X_complete)

        for i in range(n_complete):
            current_sample = X_complete[i]
            current_value = y_complete[i]
            distances = np.linalg.norm(X_complete - current_sample, axis=1)
            sorted_indices = np.argsort(distances)
            validation_indices = sorted_indices[1:k_validation + 1]

            best_l = None
            best_error = float('inf')
            best_model_info = None

            for l, model_info in models_dict[i].items():
                total_error = 0
                valid_count = 0

                for j in validation_indices:
                    if j == i:
                        continue

                    validation_sample = X_complete[j]
                    true_value = y_complete[j]

                    if model_info['type'] == 'ridge':
                        pred_value = model_info['model'].predict(validation_sample.reshape(1, -1))[0]
                    elif model_info['type'] == 'mean':
                        pred_value = model_info['value']
                    elif model_info['type'] == 'direct':
                        pred_value = model_info['value']
                    else:
                        continue

                    error = (pred_value - true_value) ** 2
                    total_error += error
                    valid_count += 1

                if valid_count > 0:
                    avg_error = total_error / valid_count
                    if avg_error < best_error:
                        best_error = avg_error
                        best_l = l
                        best_model_info = model_info

            optimal_models[i] = best_model_info

        return optimal_models

    # imputation stage
    def imputation_phase(X_incomplete, X_complete, y_complete, optimal_models, k_imputation=5):
        predictions = []

        for incomplete_sample in X_incomplete:
            distances = np.linalg.norm(X_complete - incomplete_sample, axis=1)
            sorted_indices = np.argsort(distances)
            imputation_neighbors = sorted_indices[:k_imputation]

            candidate_predictions = []
            candidate_weights = []

            for neighbor_idx in imputation_neighbors:
                model_info = optimal_models[neighbor_idx]

                if model_info['type'] == 'ridge':
                    pred_value = model_info['model'].predict(incomplete_sample.reshape(1, -1))[0]
                elif model_info['type'] == 'mean':
                    pred_value = model_info['value']
                elif model_info['type'] == 'direct':
                    pred_value = model_info['value']
                else:
                    continue

                candidate_predictions.append(pred_value)
                candidate_weights.append(1.0)

            if len(candidate_predictions) > 1:
                consensus_scores = []
                for i, pred_i in enumerate(candidate_predictions):
                    consensus = 0
                    for j, pred_j in enumerate(candidate_predictions):
                        if i != j:
                            consensus += 1.0 / (1.0 + abs(pred_i - pred_j))
                    consensus_scores.append(consensus)

                total_consensus = sum(consensus_scores)
                if total_consensus > 0:
                    candidate_weights = [score / total_consensus for score in consensus_scores]

            final_prediction = np.average(candidate_predictions, weights=candidate_weights)
            predictions.append(final_prediction)

        return predictions

    y_complete = complete_df[target_column].values
    l_values = list(range(5, min(50, len(complete_df)), 5))
    if not l_values:
        l_values = [min(5, len(complete_df))]
    print(f"Learning individual models with l_values: {l_values}")

    models_dict = learning_phase(X_complete_scaled, y_complete, l_values)
    optimal_models = select_optimal_models(models_dict, X_complete_scaled, y_complete)
    if len(X_incomplete_scaled) > 0:
        predictions = imputation_phase(X_incomplete_scaled, X_complete_scaled, y_complete, optimal_models)
        for idx, pred_value in zip(incomplete_df.index, predictions):
            df_copy.at[idx, target_column] = pred_value

    df_copy.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)