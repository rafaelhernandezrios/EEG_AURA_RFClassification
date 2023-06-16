import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
# Paso 1: Cargar el archivo CSV procesado
df = pd.read_csv('C:/Users/edgar/OneDrive/Escritorio/Ryanprojecto/CSV/csv_fusionado_processed.csv')

# Paso 2: Dividir los datos en características y etiquetas
X = df[['Mean', 'STD', 'Asymmetry']]
y = df['Label']

# Paso 3: Dividir los datos en conjuntos de entrenamiento y prueba con más aleatoriedad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Paso 4: Entrenar el clasificador Random Forest
clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Paso 5: Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Paso 6: Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud del modelo: {accuracy}")
# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.title('Matriz de Confusión')
plt.show()

with open('random_forest.pkl', 'wb') as f:
    pickle.dump(clf, f)