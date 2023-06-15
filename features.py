import pandas as pd
import numpy as np
import os

def calculate_statistics(df, original_df):
    df['Mean'] = df.mean(axis=1)
    df['STD'] = df.std(axis=1)
    df['Asymmetry'] = df.iloc[:, 0] - df.iloc[:, 2]
    df['Label'] = original_df.iloc[:, -1]  # Agregar la última columna del DataFrame original como 'Label'
    return df[['Mean', 'STD', 'Asymmetry', 'Label']]

def normalize_columns(df):
    numeric_columns = df.select_dtypes(include=np.number).columns
    normalized_df = df.copy()
    normalized_df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].min()) / (df[numeric_columns].max() - df[numeric_columns].min())
    return normalized_df

def apply_moving_mean_filter(df, window_size):
    filtered_df = df.rolling(window=window_size, min_periods=1).mean()
    return filtered_df

def normalize_and_filter_csv(input_file):
    output_file = os.path.splitext(input_file)[0] + '_processed.csv'
    window_size = 3

    df = pd.read_csv(input_file, header=None)  # Agregar el argumento 'header=None' para manejar los datos sin encabezado
    df = df.drop(0)

    normalized_df = normalize_columns(df)
    filtered_df = apply_moving_mean_filter(normalized_df, window_size)

    processed_df = calculate_statistics(filtered_df, df)
    processed_df.to_csv(output_file, index=False)
    print(f"Processed CSV file generated: {output_file}")

# Example usage
input_file = r'C:\Users\edgar\OneDrive\Escritorio\Ryanprojecto\CSV\csv_fusionado.csv'
normalize_and_filter_csv(input_file)
