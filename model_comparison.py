import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import joblib
import os
from time import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load data
def load_data():
    csv_path = os.path.join('CSV', 'csv_fusionado_processed.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"El archivo {csv_path} no existe. Asegúrate de ejecutar features.py primero.")
    
    print(f"Cargando datos desde: {csv_path}")
    data = pd.read_csv(csv_path)
    
    if 'Label' not in data.columns:
        print("Columnas disponibles:", data.columns.tolist())
        raise KeyError("La columna 'Label' no se encuentra en el CSV. Verifica el nombre de la columna.")
    
    X = data[['Mean', 'STD', 'Asymmetry']]
    y = data['Label']
    return X, y

def custom_cross_val_score(model, X, y, cv, timeout=300):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    start_time = time()
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=cv, desc="Validación cruzada")):
        if time() - start_time > timeout:
            print(f"\nTimeout alcanzado después de {fold} folds. Deteniendo validación cruzada.")
            break
            
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        with tqdm(total=1, desc=f"Entrenando fold {fold+1}/{cv}", leave=False) as pbar:
            model.fit(X_train, y_train)
            pbar.update(1)
        
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return np.array(scores)

# Train and evaluate models
def train_and_evaluate_models(X, y):
    print("\nPreprocesando datos...")
    t0 = time()
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Preprocesamiento completado en {time() - t0:.2f} segundos")
    
    # Initialize all models with optimized parameters
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        'SVM': LinearSVC(
            random_state=42,
            max_iter=1000,
            dual=False
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            max_depth=10
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5
        )
    }
    
    # Evaluate all models
    scores = {}
    for name, model in models.items():
        print(f"\nEvaluando {name}...")
        t0 = time()
        model_scores = custom_cross_val_score(model, X_scaled, y, cv=5, timeout=300)
        scores[name] = model_scores.mean() if len(model_scores) > 0 else 0
        print(f"{name} evaluado en {time() - t0:.2f} segundos")
    
    # Print results
    print("\nResultados:")
    for name, score in scores.items():
        print(f"{name} mean accuracy: {score:.4f}")
    
    # Find best model
    best_model_name = max(scores, key=scores.get)
    print(f"\n{best_model_name} tiene el mejor rendimiento. Guardando modelo...")
    
    # Train final model
    print("\nEntrenando modelo final...")
    t0 = time()
    best_model = models[best_model_name]
    
    with tqdm(total=1, desc="Entrenamiento final") as pbar:
        best_model.fit(X_scaled, y)
        pbar.update(1)
    
    print(f"Entrenamiento final completado en {time() - t0:.2f} segundos")
    
    # Save the model and scaler
    print("\nGuardando modelos...")
    with tqdm(total=2, desc="Guardando modelos") as pbar:
        model_name = best_model_name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(best_model, model_name)
        pbar.update(1)
        joblib.dump(scaler, 'scaler.pkl')
        pbar.update(1)
    
    return best_model, scaler

if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data()
    
    print("\nTraining and evaluating models...")
    best_model, scaler = train_and_evaluate_models(X, y)
    
    print("\nProcess completed!") 