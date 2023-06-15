import os
import pandas as pd

# Directorio de los archivos CSV
folder_path = 'C:/Users/edgar/OneDrive/Escritorio/Ryanprojecto/CSV'

# Obtener la lista de archivos CSV en el directorio
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Variable para almacenar el número mínimo de filas
min_rows = float('inf')

# Leer los archivos CSV y encontrar el número mínimo de filas
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, skiprows=2)
    num_rows = len(df)
    if num_rows < min_rows:
        min_rows = num_rows

# Crear una nueva carpeta para guardar los archivos procesados
output_folder = os.path.join(folder_path, 'csv_procesados')
os.makedirs(output_folder, exist_ok=True)

# Columnas a eliminar
columnas_a_eliminar = list(range(16)) + [21, 22, 29, 30, 37, 38]

# Leer nuevamente los archivos CSV, cortar las primeras dos filas, eliminar columnas y generar un archivo CSV separado para cada uno
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    output_file = os.path.join(output_folder, os.path.splitext(file)[0] + '_processed.csv')
    df = pd.read_csv(file_path, skiprows=1)
    df = df.drop(df.columns[columnas_a_eliminar], axis=1)  # Eliminar columnas
    processed_df = df.head(min_rows)
    processed_df.to_csv(output_file, index=False)
    print(f"Archivo CSV procesado generado: {output_file}")
