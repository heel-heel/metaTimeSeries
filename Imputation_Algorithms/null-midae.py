import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras import layers, models
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def process_and_fill(input_file, output_file, target_column, nonnumerical_column):
    df_copy = pd.read_csv(input_file)
    if nonnumerical_column != "None":
        df = df_copy.drop(columns=[nonnumerical_column])
    else:
        df = df_copy
    df[target_column] = df[target_column].fillna(df[target_column].mean())

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df)

    noise_factor = 0.5
    X_noisy = X_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_scaled.shape)
    X_noisy = np.clip(X_noisy, 0.0, 1.0)
    input_dim = X_scaled.shape[1]
    encoding_dim = 5

    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(X_noisy, X_scaled, epochs=100, batch_size=10, shuffle=True,
                validation_split=0.2, callbacks=[early_stopping])

    c = 10
    imputed_matrices = []
    for _ in range(c):
        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(X_noisy, X_scaled, epochs=100, batch_size=10, shuffle=True,
                    validation_split=0.2, callbacks=[early_stopping])
        imputed_matrix = autoencoder.predict(X_scaled)
        imputed_matrices.append(imputed_matrix)
    final_imputed_matrix = np.mean(imputed_matrices, axis=0)
    final_imputed_matrix = scaler.inverse_transform(final_imputed_matrix)
    target_index = df.columns.get_loc(target_column)
    df_copy[target_column] = np.where(df_copy[target_column].isnull(), final_imputed_matrix[:, target_index], df_copy[target_column])

    df_copy.to_csv(output_file, index=False)
    print(f'{output_file} has been saved.')



if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_column = sys.argv[3]
    nonnumerical_column = sys.argv[4]
    process_and_fill(input_file, output_file, target_column, nonnumerical_column)