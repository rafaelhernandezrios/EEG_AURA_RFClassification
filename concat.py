import os
import csv

def concatenate_csv_files(folder_path, output_file):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        header_written = False

        for file in csv_files:
            file_path = os.path.join(folder_path, file)

            with open(file_path, 'r') as infile:
                reader = csv.reader(infile)

                if not header_written:
                    writer.writerows(reader)
                    header_written = True
                else:
                    next(reader)
                    writer.writerows(reader)

    print(f"Archivos CSV concatenados y guardados en: {output_file}")
# Ejemplo de uso
folder_path = r'C:\Users\edgar\OneDrive\Escritorio\Ryanprojecto\CSV\csv_procesados'
output_file = r'C:\Users\edgar\OneDrive\Escritorio\Ryanprojecto\CSV\csv_fusionado.csv'
concatenate_csv_files(folder_path, output_file)
