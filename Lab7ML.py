from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import numpy as np

# Cargar los datasets (iris, wine, breast cancer de sklearn como ejemplos)
datasets = [load_iris(), load_wine(), load_breast_cancer()]
dataset_names = ["Iris", "Wine", "Breast Cancer"]

# Lista para almacenar resultados
results = []

# Valores de K a probar
k_values = [1, 3, 5, 7, 9]

# Función para imprimir resultados
def print_results(name, k, acc, cm):
    print(f"\n{name} - K: {k}")
    print(f"Accuracy: {acc}")
    print("Confusion Matrix:\n", cm)

for data, name in zip(datasets, dataset_names):
    X, y = data.data, data.target
    best_k = None
    best_accuracy = 0
    best_conf_matrix = None
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Hold-Out 70/30 estratificado
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print_results(f"{name} - Hold-Out 70/30", k, acc, cm)
        
        # Guardar mejor resultado para Hold-Out
        if acc > best_accuracy:
            best_k = k
            best_accuracy = acc
            best_conf_matrix = cm

        # 10-Fold Cross-Validation estratificado
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        acc_cv = cross_val_score(knn, X, y, cv=skf, scoring='accuracy')
        print(f"{name} - 10-Fold CV - K: {k} | Accuracy: {acc_cv.mean()}")

        # Leave-One-Out
        loo = LeaveOneOut()
        acc_loo = cross_val_score(knn, X, y, cv=loo, scoring='accuracy')
        print(f"{name} - Leave-One-Out - K: {k} | Accuracy: {acc_loo.mean()}")

    # Almacenar el mejor K y sus resultados
    results.append({
        "Dataset": name,
        "Best K": best_k,
        "Accuracy (Hold-Out)": best_accuracy,
        "Confusion Matrix (Hold-Out)": best_conf_matrix
    })

# Imprimir resultados finales
for result in results:
    print(f"\nDataset: {result['Dataset']}")
    print(f"Best K: {result['Best K']}")
    print(f"Accuracy (Hold-Out): {result['Accuracy (Hold-Out)']}")
    print("Confusion Matrix (Hold-Out):\n", result["Confusion Matrix (Hold-Out)"])
