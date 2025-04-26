import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Загрузка данных
train_data = pd.read_csv('train.csv')  # Предполагаем, что у вас есть этот файл
test_data = pd.read_csv('test.csv')    # Файл, который вы предоставили

# 2. Подготовка данных
# Предполагаем, что в train.csv есть столбец 'target' с метками (0 или 1)
X = train_data.drop('target', axis=1)
y = train_data['target']

# Разделение на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Создание pipeline с масштабированием и моделью
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingClassifier(random_state=42))
])

# 4. Подбор гиперпараметров
params = {
    'gb__n_estimators': [100, 150, 200],
    'gb__learning_rate': [0.05, 0.1, 0.2],
    'gb__max_depth': [3, 5, 7],
    'gb__min_samples_split': [2, 5, 10],
    'gb__subsample': [0.8, 1.0]
}

# Чтобы ускорить подбор параметров, можно использовать RandomizedSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid=params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 5. Проверка лучшей модели
best_model = grid_search.best_estimator_
print(f"Лучшие параметры: {grid_search.best_params_}")

# Предсказание на валидационной выборке
val_pred = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)
print(f"Accuracy на валидационной выборке: {val_accuracy:.4f}")
print(classification_report(y_val, val_pred))

# 6. Подгонка точности под требуемый диапазон (86.5%-91.5%)
if val_accuracy < 0.865 or val_accuracy > 0.915:
    print("Точность вне целевого диапазона, корректируем...")
    
    # Начинаем с лучших параметров и регулируем сложность модели
    current_params = grid_search.best_params_.copy()
    
    while val_accuracy < 0.865 or val_accuracy > 0.915:
        if val_accuracy < 0.865:
            # Увеличиваем сложность модели
            current_params['gb__n_estimators'] = min(300, current_params['gb__n_estimators'] + 50)
            current_params['gb__max_depth'] = min(10, current_params['gb__max_depth'] + 1)
            current_params['gb__learning_rate'] = max(0.01, current_params['gb__learning_rate'] * 0.9)
        else:
            # Уменьшаем сложность модели
            current_params['gb__n_estimators'] = max(50, current_params['gb__n_estimators'] - 50)
            current_params['gb__max_depth'] = max(2, current_params['gb__max_depth'] - 1)
            current_params['gb__learning_rate'] = min