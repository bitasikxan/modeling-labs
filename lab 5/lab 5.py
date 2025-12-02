import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


path = "C:\\Users\\legen\\Desktop\\model\\lab 5\\vowel.csv"
dataset = pd.read_csv(path)

print("Статистичні дані:")
print(dataset.describe())

# перевірка на дублікати - все по нулях, коригування не потрібне
print("")
print(f"Дублікатів: {dataset.duplicated().sum()}")
print(f"Пропусків: {dataset.isnull().sum().sum()}")
print("К-сть записів: ", dataset.count().max())

print('\nРозподіл класів:\n', dataset.groupby('Class').size())

# матриця графіків розсіювання
classes = dataset['Class'].unique()

cmap = matplotlib.colormaps['jet']
colors_array = cmap(np.linspace(0, 1, len(classes)))
color_wheel = {cls: colors_array[i] for i, cls in enumerate(classes)}
dataset_sampled = dataset.sample(frac=0.5, random_state=42)
colors = dataset_sampled["Class"].map(lambda x: color_wheel.get(x))

scatter_matrix(dataset_sampled, c=colors, figsize=(9, 9), s=60)
plt.show()

# X - ознаки, y - класи
X = dataset.drop('Class', axis=1)
y = dataset['Class']

# кодування класів
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"\nРозмір навчальної вибірки: {X_train.shape[0]} записів")
print(f"Розмір тестової вибірки: {X_test.shape[0]} записів")


# === ФІКСОВАНІ ПАРАМЕТРИ, БЕЗ МАСШТАБУВАННЯ ===
grid = GridSearchCV(SVC(kernel='rbf'), {'C': [1], 'gamma': [0.1]}, cv=5, scoring='accuracy')

print("Починаю навчання...")
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

print("\nЗвіт по класифікації:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm=confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()
plt.show()

# === ФІКСОВАНІ ПАРАМЕТРИ, МАСШТАБУВАННЯ ===
# об'єднання скейлінгу та класифікатора в одне ціле, щоб уникнути витоку даних
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Нормалізація (Z-score)
    ('svm', SVC(kernel='rbf'))          # Класифікатор
])

grid = GridSearchCV(pipeline, {'svm__C': [1], 'svm__gamma': [0.1]}, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

print("\nЗвіт по класифікації з масштабуванням:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === ПЛАВАЮЧІ ПАРАМЕТРИ, МАСШТАБУВАННЯ ===
param_grid = {
    'svm__C': [0.1, 1, 10, 100],        # штраф за помилку
    'svm__gamma': ['scale', 0.1, 0.01, 0.001], # вплив точок
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Найкращі параметри: {grid.best_params_}")
print(f"Найкраща точність на крос-валідації: {grid.best_score_:.2f}")

y_pred = grid.predict(X_test)

print("\nЗвіт по класифікації з найкращими параметрами:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === RANDOM FOREST (ДЛЯ ПОРІВНЯННЯ) ===
print("\n--- Порівняння з Random Forest ---")

rf_params = {
    'n_estimators': [50, 100, 200],   # к-сть дерев
    'max_depth': [None, 10, 20],      # глибина (щоб не перевчився)
    'min_samples_split': [2, 5]       # мінімальна к-сть точок для поділу
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy', n_jobs=-1)

print("Навчання Random Forest...")
grid_rf.fit(X_train, y_train)

print(f"Найкращі параметри RF: {grid_rf.best_params_}")
print(f"Точність RF на крос-валідації: {grid_rf.best_score_:.2f}")

y_pred_rf = grid_rf.predict(X_test)

print("\nЗвіт по Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=[str(c) for c in le.classes_]))


def get_euclidean_distance(row1, row2):
    distance = np.sqrt(np.sum((row1 - row2) ** 2))
    return distance


print("\n=== Ручна реалізація пошуку сусіда ===")

X_test_np = X_test.values
X_train_np = X_train.values
y_test_np = y_test
y_train_np = y_train

results = []
unique_classes = np.unique(y_test)

for class_label in unique_classes:
    indices = np.where(y_test_np == class_label)[0]
    selected_indices = indices[:3]
    for test_idx in selected_indices:
        test_point = X_test_np[test_idx]
        actual_class_code = y_test_np[test_idx]

        # --- пошук найближчого сусіда ---
        min_dist = float('inf')
        nearest_neighbor_class_code = -1

        # перебираємо весь тренувальний набір
        for train_idx, train_point in enumerate(X_train_np):
            dist = get_euclidean_distance(test_point, train_point)

            if dist < min_dist:
                min_dist = dist
                nearest_neighbor_class_code = y_train_np[train_idx]


        actual_name = le.inverse_transform([actual_class_code])[0]
        neighbor_name = le.inverse_transform([nearest_neighbor_class_code])[0]

        is_match = (actual_class_code == nearest_neighbor_class_code)
        results.append(is_match)

        status = "+" if is_match else "-"
        print(f"Тест: {actual_name:<5} -> Сусід: {neighbor_name:<5} (Дист: {min_dist:.4f}) | {status}")

accuracy_manual = (sum(results) / len(results)) * 100
print(f"\nТочність власного методу (на вибірці): {accuracy_manual:.2f}%")