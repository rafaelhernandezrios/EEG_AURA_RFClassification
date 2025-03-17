import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_bypred
import joblib
import time
from datetime import datetime
import os
from collections import deque

def initialize_lsl():
    """Inicializa la conexión LSL con AURA"""
    print("Buscando stream AURA_Filtered...")
    streams = resolve_bypred("name='AURA_Filtered'")
    
    if not streams:
        print("No se encontró el stream AURA_Filtered")
        return None
    
    inlet = StreamInlet(streams[0])
    print("Conexión establecida con AURA_Filtered")
    return inlet

def process_window(data_buffer):
    """Procesa una ventana de datos aplicando las mismas transformaciones que en el entrenamiento"""
    # Convertir el buffer a DataFrame
    df = pd.DataFrame(data_buffer, columns=[f'Channel_{i}' for i in range(len(data_buffer[0]))])
    
    # Normalizar
    numeric_columns = df.select_dtypes(include=np.number).columns
    df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].min()) / (df[numeric_columns].max() - df[numeric_columns].min())
    
    # Aplicar filtro de media móvil
    window_size = 3
    df = df.rolling(window=window_size, min_periods=1).mean()
    
    # Calcular características
    features = pd.DataFrame({
        'Mean': df.mean(axis=1),
        'STD': df.std(axis=1),
        'Asymmetry': df.iloc[:, 0] - df.iloc[:, 2]  # Asumiendo que las columnas 0 y 2 son las correctas
    })
    
    return features.iloc[-1:].values  # Retornar solo la última fila

def main():
    # Cargar el modelo y el scaler
    print("Cargando modelo...")
    model_files = [f for f in os.listdir('.') if f.endswith('.pkl') and f != 'scaler.pkl']
    if not model_files:
        print("No se encontró ningún modelo entrenado (.pkl)")
        return
    
    model = joblib.load(model_files[0])
    scaler = joblib.load('scaler.pkl')
    print(f"Modelo cargado: {model_files[0]}")
    
    # Inicializar LSL
    inlet = initialize_lsl()
    if inlet is None:
        return
    
    # Buffer circular para almacenar las últimas muestras
    window_size = 150  # 1 segundo de datos a 150Hz
    data_buffer = deque(maxlen=window_size)
    
    try:
        print("Iniciando clasificación en tiempo real...")
        print("Presiona Ctrl+C para detener")
        
        # Llenar el buffer inicial
        print("Recolectando datos iniciales...")
        while len(data_buffer) < window_size:
            sample, timestamp = inlet.pull_sample()
            if sample is not None:
                data_buffer.append(sample)
        
        print("Comenzando clasificación...")
        while True:
            # Obtener nueva muestra
            sample, timestamp = inlet.pull_sample(timeout=0.0)
            if sample is None:
                time.sleep(0.001)  # Pequeña pausa si no hay datos
                continue
            
            # Agregar muestra al buffer
            data_buffer.append(sample)
            
            # Procesar datos cuando el buffer está lleno
            if len(data_buffer) == window_size:
                # Convertir deque a array para procesamiento
                data_array = np.array(data_buffer)
                
                # Procesar datos
                features = process_window(data_array)
                
                # Escalar características
                features_scaled = scaler.transform(features)
                
                # Realizar predicción
                prediction = model.predict(features_scaled)
                
                # Imprimir resultado con timestamp
                timestamp_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp_str}] Estado emocional detectado: {prediction[0]}")
                
                # Pequeña pausa para no saturar la consola
                time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nDeteniendo la clasificación...")
    except Exception as e:
        print(f"\nError durante la clasificación: {e}")
    finally:
        print("Sesión de AURA finalizada")

if __name__ == "__main__":
    main() 