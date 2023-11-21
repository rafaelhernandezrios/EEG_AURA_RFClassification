import os
import pandas as pd

def get_csv_dimensions(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    file_dimensions = {}

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        rows, columns = df.shape
        file_dimensions[file] = (rows, columns)

    return file_dimensions

# Ejemplo de uso
folder_path = r'C:\Users\edgar\OneDrive\Escritorio\Ryanprojecto\CSV\csv_procesados'
dimensions = get_csv_dimensions(folder_path)

for file, (rows, columns) in dimensions.items():
    print(f"Tamaño de {file}: {rows} filas, {columns} columnas")
