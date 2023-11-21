# RFEEG: EEG Data Processing and Analysis

## Descripción
RFEEG es un proyecto dedicado al procesamiento y análisis de datos de electroencefalograma (EEG). Utiliza una serie de scripts en Python para preparar, manipular y analizar datos de EEG, incluyendo la implementación de un modelo de Random Forest para la clasificación o análisis.
## Instalación

Para instalar y ejecutar este proyecto, siga estos pasos:

Clonar el repositorio
git clone https://github.com/edgarhernandez94/RFEEG.git

Navegar al directorio del proyecto
cd RFEEG


## Uso
Una vez instalado, puede ejecutar el proyecto con el siguiente comando:
python main.py

Este comando ejecutará el script `main.py`, que es el punto de entrada principal del proyecto. Asegúrese de haber configurado su entorno con las dependencias necesarias y haber colocado los datos EEG en el formato y ubicación correctos, como se describe en la documentación.

## Scripts

### `cortarcsv.py`
Procesa archivos CSV. Principales funciones:
- Leer y procesar archivos CSV.
- Determinar el número mínimo de filas en los archivos.
- Eliminar columnas específicas.
- Guardar archivos CSV procesados.

### `concat.py`
Concatena múltiples archivos CSV en uno solo. Características:
- Concatena archivos en un directorio especificado.
- Mantiene la cabecera del primer archivo, omite las demás.
- Guarda el archivo de salida en una ubicación definida.

### `features.py`
Procesamiento avanzado de datos CSV. Funciones clave:
- Normalizar columnas numéricas.
- Aplicar filtro de media móvil.
- Calcular estadísticas como media, desviación estándar y asimetría.
- Añadir etiquetas y guardar en un nuevo archivo CSV.

### `RF.py`
Implementa un clasificador de Random Forest. Incluye:
- Carga y preparación de datos.
- División de datos en conjuntos de entrenamiento y prueba.
- Entrenamiento del clasificador Random Forest.
- Evaluación del modelo con exactitud y matriz de confusión.
- Guardar el modelo entrenado.

## Contribuir

Para contribuir al proyecto, por favor haga fork del repositorio, cree una rama con sus cambios y envíe un pull request para revisión.




