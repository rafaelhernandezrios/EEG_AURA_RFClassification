import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Paso 1: Cargar el archivo CSV procesado
df = pd.read_csv('./CSV/csv_fusionado_processed.csv')

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

# Paso 6: Calcular el accuracy del modelo
accuracy = clf.score(X_test, y_test)
print(f"Exactitud del modelo: {accuracy}")

# Paso 7: Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Paso 8: Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.title('Matriz de Confusión')
plt.show()

# Paso 9: Visualizar la gráfica de precisión para diferentes hiperparámetros
n_estimators = [50, 100, 150, 200]
max_depth = [3, 5, 7, 9]
train_accuracy = []
val_accuracy = []

for n in n_estimators:
    for depth in max_depth:
        clf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_test, y_test)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)

plt.figure(figsize=(8, 6))
plt.plot(range(len(train_accuracy)), train_accuracy, label='Precisión en entrenamiento')
plt.plot(range(len(val_accuracy)), val_accuracy, label='Precisión en validación')
plt.xticks(range(len(n_estimators)*len(max_depth)), [f'{n}_{depth}' for n in n_estimators for depth in max_depth])
plt.xlabel('Valores de hiperparámetros')
plt.ylabel('Precisión')
plt.title('Precisión del modelo para diferentes hiperparámetros')
plt.legend()
plt.show()
