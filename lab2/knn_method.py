import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from statsmodels.nonparametric.smoothers_lowess import lowess


class CustomKNN(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5, window_size="fixed", metric="euclidean", kernel="uniform", weights="uniform",
                 prior_weights=None, distance_threshold=None):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.metric = metric
        self.kernel = kernel
        self.weights = weights
        self.prior_weights = prior_weights
        self.distance_threshold = distance_threshold

    def fit(self, X, y):
        # сохраним обучающие данные
        self.X_train = X
        self.y_train = y
        self.label_encoder = LabelEncoder()
        self.y_train_encoded = self.label_encoder.fit_transform(y)
        if self.prior_weights is None:
            self.prior_weights = np.ones(len(y))
        return self

    def _compute_weights(self, distances):
        if self.kernel == "uniform":
            return np.ones_like(distances)
        elif self.kernel == "gaussian":
            return np.exp(-distances ** 2 / 2)
        elif self.kernel == "exponential":
            return np.exp(-distances)
        elif self.kernel == "laplace":
            return np.exp(-np.abs(distances))

    def predict(self, X):
        neighbors = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(self.X_train)), metric=self.metric).fit(
            self.X_train)
        distances, indices = neighbors.kneighbors(X)

        predictions = []
        for i in range(len(X)):
            if self.window_size == "variable":
                #distance_threshold задан используем его для фильтрации по расстоянию
                if self.distance_threshold is not None:
                    variable_indices = indices[i][distances[i] <= self.distance_threshold]
                    variable_distances = distances[i][distances[i] <= self.distance_threshold]
                else:
                    # distance_threshold не задан используем квантиль
                    threshold = np.percentile(distances[i], 90)
                    variable_indices = indices[i][distances[i] <= threshold]
                    variable_distances = distances[i][distances[i] <= threshold]
            else:
                variable_indices = indices[i]
                variable_distances = distances[i]

            weights = self._compute_weights(variable_distances)

            if self.weights == "uniform":
                prediction = np.bincount(self.y_train_encoded[variable_indices]).argmax()
            elif self.weights == "distance":
                weighted_votes = np.zeros(len(np.unique(self.y_train_encoded)))
                for j, idx in enumerate(variable_indices):
                    weighted_votes[self.y_train_encoded[idx]] += weights[j] * self.prior_weights[idx]
                prediction = np.argmax(weighted_votes)

            predictions.append(prediction)

        return self.label_encoder.inverse_transform(predictions)

data = pd.read_csv("normed_data.csv", delimiter=",")
price_category_labels = ["Бедный студент (минимум)", "Cтудент со стипендией (чуть ниже среднего)",
                         "Стажер в Яндексе (средняя)", "Middle Программист (выше среднего)",
                         "Senior Developer (очень высокая)"]
quantil = [0, 0.15, 0.35, 0.65, 0.85, 1]

# Добавление ценовой категории
data["Ценовая категория"] = pd.qcut(data['Стоимость (рубли)'], q=quantil, labels=price_category_labels)

selected_features = ["общая (м²)", "Расстояние до метро (мин)", "Тип квартиры", "отделка"]
X = data[selected_features]

X = X.values  # Преобразование в numpy-массив
# X = data.drop(columns=["URL", "Стоимость (рубли)", "Станция метро", "Ценовая категория"]).values
y = data["Ценовая категория"].values

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# наши гиперпараметры
param_grid = {
    'n_neighbors': [1, 3, 5, 11, 14],
    'metric': ['euclidean', 'manhattan', 'cosine'],
    'kernel': ['uniform', 'gaussian', 'exponential', 'laplace'],
    'weights': ['uniform', 'distance'],
    'window_size': ['fixed', 'variable']
}

# Подбор гиперпараметров с помощью GridSearchCV
grid_search = GridSearchCV(CustomKNN(), param_grid, cv=20, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Результаты перебора гиперпараметров:")
results = grid_search.cv_results_
for i in range(len(results['params'])):
    print(f"Combination {i + 1}: {results['params'][i]}")
    print(f"Mean Test Accuracy: {results['mean_test_score'][i]:.4f}")
    print(f"Std Dev of Accuracy: {results['std_test_score'][i]:.4f}")  # разброс точности для наших фолдов
    print("-" * 50)

# Лучшая комбинация гиперпараметров
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")
print(f"Best Accuracy on Validation : {grid_search.best_score_:.4f}")

# на валидационной выборке
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# на тестовой выборке
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# LOWESS
frac_value = 0.2
X_combined = np.vstack((X_train, X_val, X_test))
X_combined_lowess = X_combined.copy()

for feature_idx in range(X_combined.shape[1]):
    smoothed = lowess(X_combined[:, feature_idx], np.arange(len(X_combined)), frac=frac_value)[:, 1]
    X_combined_lowess[:, feature_idx] = smoothed

# разделение сглаженных данных
X_train_lowess = X_combined_lowess[:len(X_train)]
X_val_lowess = X_combined_lowess[len(X_train):len(X_train) + len(X_val)]
X_test_lowess = X_combined_lowess[len(X_train) + len(X_val):]

# на модели после применения LOWESS
model_with_lowess = CustomKNN(**best_params)
model_with_lowess.fit(X_train_lowess, y_train)

# на валидационной выборке
y_val_pred_lowess = model_with_lowess.predict(X_val_lowess)
val_accuracy_lowess = accuracy_score(y_val, y_val_pred_lowess)
print(f"Validation Accuracy After LOWESS: {val_accuracy_lowess:.4f}")

# на тестовой выборке
y_test_pred_lowess = model_with_lowess.predict(X_test_lowess)
test_accuracy_lowess = accuracy_score(y_test, y_test_pred_lowess)
print(f"Test Accuracy After LOWESS: {test_accuracy_lowess:.4f}")

neighbors_range = [3, 7, 10, 20]
train_errors = []
val_errors = []
test_errors = []

for n_neighbors in neighbors_range:
    model = CustomKNN(n_neighbors=n_neighbors, **{k: v for k, v in best_params.items() if k != 'n_neighbors'})
    model.fit(X_train_lowess, y_train)

    y_train_pred = model.predict(X_train_lowess)
    y_val_pred = model.predict(X_val_lowess)
    y_test_pred = model.predict(X_test_lowess)

    train_error = mean_squared_error(model.label_encoder.transform(y_train),
                                     model.label_encoder.transform(y_train_pred))
    val_error = mean_squared_error(model.label_encoder.transform(y_val), model.label_encoder.transform(y_val_pred))
    test_error = mean_squared_error(model.label_encoder.transform(y_test), model.label_encoder.transform(y_test_pred))

    train_errors.append(train_error)
    val_errors.append(val_error)
    test_errors.append(test_error)

plt.figure(figsize=(10, 6))
plt.plot(neighbors_range, train_errors, label='Train Error', marker='o', color='blue')
plt.plot(neighbors_range, val_errors, label='Validation Error', marker='s', color='green')
plt.plot(neighbors_range, test_errors, label='Test Error', marker='x', color='red')
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean Squared Error')
plt.title('Error vs Number of Neighbors with LOWESS')
plt.legend()
plt.grid(True)
plt.show()
