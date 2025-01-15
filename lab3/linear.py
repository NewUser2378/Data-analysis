import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

data = pd.read_csv("normed_data.csv", delimiter=",")
price_category_labels = ["не выше рынка", "выше рынка"]
quantil = [0, 0.65, 1]
data["Ценовая категория"] = pd.qcut(data['Стоимость (рубли)'], q=quantil, labels=price_category_labels)

data["Ценовая категория"] = data["Ценовая категория"].map({"не выше рынка": -1, "выше рынка": 1})

data["Ценовая категория"] = data["Ценовая категория"].astype(int)
selected_features = ["общая (м²)", "Расстояние до метро (мин)", "Тип квартиры", "отделка"]
X = data[selected_features].values
y = data["Ценовая категория"].values
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_val_bias = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])


def ridge_regression_custom(X, y, lambda_reg):
    """
    X: Матрица признаков (с добавлением столбца единиц для смещения)
    y: Целевой вектор
    lambda_reg: Параметр регуляризации
    return Вектор коэффициентов (включая смещение)
    """
    n, m = X.shape
    I = np.eye(m)  # 1ая матрица для регуляризации
    I[0, 0] = 0  # не хотим регуляризовывать смещение,поэтому занулим 1 элемент w0
    # формула из лекции (X.T @ X + lambda * I)^-1 @ X.T @ y
    XtX = X.T @ X
    XtY = X.T @ y

    w = np.linalg.inv(XtX + lambda_reg * I) @ XtY
    return w


def predict_custom(X, w):
    """
    X: Матрица признаков (с добавленным столбцом единиц для смещения)
    w: Вектор коэффициентов
    return: Прогнозируемые значения
    """
    return X @ w


lambda_reg = 1.0
w_custom = ridge_regression_custom(X_train_bias, y_train, lambda_reg)
y_pred_test_custom = predict_custom(X_test_bias, w_custom)

# прогнозы в метки классов (+1 или -1) на основе знака
y_pred_test_class = np.sign(y_pred_test_custom)

accuracy_custom = np.mean(y_pred_test_class == y_test)
print(f'Точность на тестовых данных для матричного: {accuracy_custom * 100:.2f}%')


class LinearClassifierGD(BaseEstimator, ClassifierMixin):

    def __init__(self, risk="squared", alpha=0.5, lambda_reg=1, learning_rate=0.01, max_iter=1000):
        """
        risk: ("squared", "hinge", "exponential").
        alpha: параметр Elastic Net регуляризации (0 = чистый Ridge, 1 = чистый Lasso).
        lambda_reg: коэф регуляризации
        learning_rate: скорость
        max_iter: число итераций
        """
        self.risk = risk
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.w = None
        self.train_losses = []
        self.test_accuracies = []

    def _margin(self, X, y):
        """
        как в лекции смотрим на наш предсказанный класс и отступ считаем как произведение чтобы проверять совпадение знака
        получим что отступ положителен если правильно классифицировали то есть знаки совпали
        ну и так считаем по всем признакам
        хотим минимизировать отступ также из-за сдвига учтем и регурялизацию
        M(xi,yi) = yi(w.T xi + w0)
        """
        return y * (X @ self.w)

    def _emperial_risk_grad(self, X, y):
        """
        Считаем функцию потерь J(w) = R(w) + a*P(w)
        R(w) -- наш риск
         a-- коэф регурялизации
         P(w) -- регуляризация
         считаем градиент как сумму градиентов от слагаемых
        в этом методе будем считать градиенты только от рисков

        1) квадратичный: R(w) = 1/n (Сумма по i от макс(0, 1-M(xi,yi))^2)
        если M(xi,yi) >= 0 то 0
        градиент по w = -2/n Сумма по i от макс(0, 1-M(xi,yi))xi,yi

        2) линейный риск

        R(w) = 1/n(Сумма по i от макс(0, 1-M(xi,yi)))
        если M(xi,yi) >= 0 то 0
        градиент по w = -1/n Сумма по i от макс(0, 1-M(xi,yi))


        3) экспоненциальный

        R(w) = 1/n(Сумма по i e^(-M(xi,yi)))
        градиент по w = 1/n(Сумма по i e^(-M(xi,yi))yixi)
        """
        margins = self._margin(X, y)

        if self.risk == "squared":
            # квадратичный (1 - M)^2, если M < 1; иначе 0
            loss = np.where(margins < 1, (1 - margins) ** 2, 0)
            # reshape(-1, 1) чтобы преобразовать одномерный массив вектор столбец
            grad = np.where(margins < 1, -2 * (1 - margins) * y, 0).reshape(-1, 1) * X
        elif self.risk == "hinge":
            # линейный  max(0, 1 - M)
            loss = np.maximum(0, 1 - margins)
            grad = np.where(margins < 1, -y, 0).reshape(-1, 1) * X
        elif self.risk == "exponential":
            # экспоненциальный exp(-M)
            loss = np.exp(-margins)
            grad = -np.exp(-margins).reshape(-1, 1) * y.reshape(-1, 1) * X
        else:
            raise ValueError("Unsupported risk type. Choose from 'squared', 'hinge', 'exponential'.")

            # cреднее  по всем примерам
        risk_value = np.mean(loss)
        gradient = np.mean(grad, axis=0)

        return risk_value, gradient

    def _elastic_grad(self):
        """
        считаем второе слагаемое нашей суммы

        P(w) = k ||w||1 + (1-k) (||w||2)^2

        производная по w у ||w||1 это сигнум(w), 0 если 0,1 если > 0, -1 если меньше
        у (||w||2)^2 получим 2w
        в итоге домножаем на коэф регуляризации и   (k sign(w) + 2(1-k)w)

        """
        l1_grad = self.alpha * np.sign(self.w)
        l2_grad = 2 * (1 - self.alpha) * self.w
        grad = self.lambda_reg * (l1_grad + l2_grad)
        return grad

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)

        for i in range(self.max_iter):
            risk_val, grad_risk = self._emperial_risk_grad(X_train, y_train)
            grad_pen = self._elastic_grad()
            total_grad = grad_risk + grad_pen
            self.w -= self.learning_rate * total_grad
            self.train_losses.append(risk_val)
            if X_test is not None and y_test is not None:
                test_accuracy = self.score(X_test, y_test)
                self.test_accuracies.append(test_accuracy)

    def predict(self, X):
        return np.sign(X @ self.w)

    def score(self, X, y):
        """
        Рассчитывает точность модели на тестовых данных.
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy


classifier = LinearClassifierGD(
    risk="hinge", alpha=0.5, lambda_reg=0.1, learning_rate=0.01, max_iter=1000
)
classifier.fit(X_train_bias, y_train)
y_test_pred = classifier.predict(X_test_bias)
test_accuracy = np.mean(y_test_pred == y_test)
print(f"Точность на тестовых данных: {test_accuracy * 100:.2f}%")

import numpy as np

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np

class SVMClassifierGD(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='linear', C=1.0, learning_rate=0.01, max_iter=1000):
        self.kernel = kernel
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.test_accuracies = []

    def _kernel(self, X1, X2):
        """Вычисление ядра"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                self.gamma = 1 / X1.shape[1]
            elif self.gamma == 'auto':
                self.gamma = 1 / X1.shape[0]
            sq_dists = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError("Поддерживаются только 'linear' и 'rbf' ядра.")

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """Обучение модели методом градиентного спуска."""
        self.X_train = X_train
        self.y_train = y_train
        n_samples, n_features = X_train.shape

        self.w = np.zeros(n_features)
        self.b = 0


        for epoch in range(self.max_iter):
            for i in range(n_samples):

                prediction = np.dot(X_train[i], self.w) + self.b

                if y_train[i] * prediction < 1:
                    gradient_w = self.w - self.C * y_train[i] * X_train[i]
                    gradient_b = -self.C * y_train[i]
                else:
                    gradient_w = self.w
                    gradient_b = 0

                self.w -= self.learning_rate * gradient_w
                self.b -= self.learning_rate * gradient_b

            if X_test is not None and y_test is not None:
                test_accuracy = self.score(X_test, y_test)
                self.test_accuracies.append(test_accuracy)

        return self

    def predict(self, X):
        """Предсказание класса для новых данных."""
        return np.sign(np.dot(X, self.w) + self.b)

    def score(self, X_test, y_test):
        """Оценка точности на тестовых данных."""
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)

    def _hinge_loss(self, X, y):
        """
        Вычисление потерь по функции хинжа для SVM.
        :param X: Признаки (матрица).
        :param y: Истинные метки (вектор).
        :return: Сумма потерь.
        """
        margins = 1 - y * (np.dot(X, self.w) + self.b)
        margins[margins < 0] = 0
        return np.mean(margins) + 0.5 * np.dot(self.w, self.w)

    def get_params(self, deep=True):
        """Возвращает параметры модели."""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter
        }

    def set_params(self, **params):
        """Устанавливает параметры модели."""
        for key, value in params.items():
            setattr(self, key, value)
        return self




svm_classifier = SVMClassifierGD(kernel="rbf", C=1.0, learning_rate=0.01, max_iter=1000)
svm_classifier.fit(X_train_bias, y_train)
y_pred_svm = svm_classifier.predict(X_test_bias)
test_accuracy_svm = np.mean(y_pred_svm == y_test)
print(f"Точность на тестовых данных для SVM: {test_accuracy_svm * 100:.2f}%")

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

param_grid0 = {

    'risk': ["squared", "hinge", "exponential"],
    'alpha': [0, 0.5, 0.75, 1],
    'lambda_reg': [0.25, 0.5, 1],
    'learning_rate': [0.01, 0.1],
    'max_iter': [500, 1000]
}

param_grid = {
    'kernel': ['linear',  'rbf'],
    'degree': [2, 3],
    'C': [0.1, 1.0, 10.0],
    'learning_rate': [ 0.01, 0.1],
    'max_iter': [500, 1000]
}

base_classifaier = LinearClassifierGD()
grid_search_linear = GridSearchCV(estimator=base_classifaier, param_grid=param_grid0, cv=5, n_jobs=1)
grid_search_linear.fit(X_train_bias, y_train)
print(f"Лучшие параметры: {grid_search_linear.best_params_}")

svm_classifier = SVMClassifierGD(kernel='rbf', C=1.0, learning_rate=0.01, max_iter=1000)
grid_search_svm = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search_svm.fit(X_train_bias, y_train)

print(f"Лучшие параметры: {grid_search_svm.best_params_}")

best_svm_classifier = grid_search_svm.best_estimator_

# оценка точности на тестовых данных
y_pred_svm = best_svm_classifier.predict(X_test_bias)
test_accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Точность на тестовых данных для SVM: {test_accuracy_svm * 100:.2f}%")

# линейный классификатор
linear_classifier = LinearClassifierGD(risk="hinge", alpha=0.5, lambda_reg=0.1, learning_rate=0.01, max_iter=1000)
linear_classifier.fit(X_train_bias, y_train, X_test=X_test_bias, y_test=y_test)

# SVM классификатор

# SVM классификатор
svm_classifier = SVMClassifierGD(kernel='rbf', C=1.0, learning_rate=0.01, max_iter=1000)
svm_classifier.fit(X_train_bias, y_train)  # Убрали X_test и y_test
y_pred_svm = svm_classifier.predict(X_test_bias)  # Оценка точности на тесте
test_accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Точность на тестовых данных для SVM: {test_accuracy_svm * 100:.2f}%")


def calculate_confidence_interval(accuracy, n_samples, confidence_level=0.95):
    z = norm.ppf(1 - (1 - confidence_level) / 2)  # квантиль
    ci = z * np.sqrt((accuracy * (1 - accuracy)) / n_samples)
    return ci


# доверительные интервалы для точности
linear_test_ci = [calculate_confidence_interval(acc, len(y_test)) for acc in linear_classifier.test_accuracies]
svm_test_ci = [calculate_confidence_interval(acc, len(y_test)) for acc in svm_classifier.test_accuracies]

# целевая функции на тестовом (hinge loss)
linear_test_loss = linear_classifier._emperial_risk_grad(X_test_bias, y_test)[0]
svm_test_loss = svm_classifier._hinge_loss(X_test_bias, y_test)

# ---ДОБАВИЛ  ---
lambda_values = np.logspace(-4, 4, 50)
train_errors = []
val_errors = []

for lambda_reg in lambda_values:
    w = ridge_regression_custom(X_train_bias, y_train, lambda_reg)

    # Прогнозы на обучающих и валидационных данных
    y_train_pred = predict_custom(X_train_bias, w)
    y_val_pred = predict_custom(X_val_bias, w)

    # Расчёт эмпирического риска
    train_error = np.mean((y_train - y_train_pred) ** 2)
    val_error = np.mean((y_val - y_val_pred) ** 2)

    train_errors.append(train_error)
    val_errors.append(val_error)

# ДОБАВИЛ ______________________________
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, train_errors, label="Ошибка на тренировочных данных", color="blue")
plt.plot(lambda_values, val_errors, label="Ошибка на валидационных данных", color="orange")
plt.xscale("log")  # Логарифмическая шкала для λ
plt.xlabel("Параметр регуляризации λ")
plt.ylabel("Эмпирический риск (MSE)")
plt.title("Кривая обучения для матричного метода")
plt.legend()
plt.grid()
plt.show()

# линейная классификация
if hasattr(linear_classifier, 'train_losses'):
    plt.plot(linear_classifier.train_losses, label="Эмпирический риск (линейная классификация)", color="blue")
if hasattr(linear_classifier, 'test_losses'):
    plt.plot(linear_classifier.test_losses, label="Целевая функция (линейная классификация, тест)", linestyle="--",
             color="blue")

# SVM
if hasattr(svm_classifier, 'train_losses'):
    plt.plot(svm_classifier.train_losses, label="Эмпирический риск (SVM)", color="green")
if hasattr(svm_classifier, 'test_losses'):
    plt.plot(svm_classifier.test_losses, label="Целевая функция (SVM, тест)", linestyle="--", color="green")

plt.xlabel("Итерации")
plt.ylabel("Значение функции ошибки")
plt.title("Кривая обучения (эмпирический риск и целевая функция)")
plt.legend()
plt.grid()
plt.show()

# --- точность с доверительными интервалами ---
plt.figure(figsize=(12, 6))  # новое окно для второго графика

# линейная классификация
if hasattr(linear_classifier, 'test_accuracies'):
    plt.plot(linear_classifier.test_accuracies, label="Линейная классификация (точность на тесте)", color="blue")
    if hasattr(linear_classifier, 'test_accuracies'):
        linear_test_ci = [
            calculate_confidence_interval(acc, len(y_test)) for acc in linear_classifier.test_accuracies
        ]
        plt.fill_between(
            range(len(linear_classifier.test_accuracies)),
            np.array(linear_classifier.test_accuracies) - np.array(linear_test_ci),
            np.array(linear_classifier.test_accuracies) + np.array(linear_test_ci),
            color="blue", alpha=0.2, label="Доверительный интервал (линейная классификация)"
        )

# SVM
if hasattr(svm_classifier, 'test_accuracies'):
    plt.plot(svm_classifier.test_accuracies, label="SVM (точность на тесте)", color="green")
    if hasattr(svm_classifier, 'test_accuracies'):
        svm_test_ci = [
            calculate_confidence_interval(acc, len(y_test)) for acc in svm_classifier.test_accuracies
        ]
        plt.fill_between(
            range(len(svm_classifier.test_accuracies)),
            np.array(svm_classifier.test_accuracies) - np.array(svm_test_ci),
            np.array(svm_classifier.test_accuracies) + np.array(svm_test_ci),
            color="green", alpha=0.2, label="Доверительный интервал (SVM)"
        )

# горизонтальные линии для целевой функции
linear_test_loss = (
    linear_classifier._emperial_risk_grad(X_test_bias, y_test)[0]
    if hasattr(linear_classifier, '_emperial_risk_grad')
    else None
)
if linear_test_loss is not None:
    plt.axhline(y=linear_test_loss, color="blue", linestyle="--", label="Целевая функция (линейная классификация)")

svm_test_loss = (
    svm_classifier._hinge_loss(X_test_bias, y_test)
    if hasattr(svm_classifier, '_hinge_loss')
    else None
)
if svm_test_loss is not None:
    plt.axhline(y=svm_test_loss, color="green", linestyle="--", label="Целевая функция (SVM)")

plt.xlabel("Итерации")
plt.ylabel("Точность")
plt.title("Точность с доверительными интервалами")
plt.legend()
plt.grid()
plt.show()
