import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("normed_data.csv", delimiter=",")
price_category_labels = [
    "Бедный студент (минимум)",
    "Cтудент со стипендией (чуть ниже среднего)",
    "Стажер в Яндексе (средняя)",
    "Middle Программист (выше среднего)",
    "Senior Developer (очень высокая)"
]
quantil = [0, 0.15, 0.35, 0.65, 0.85, 1]

data["Ценовая категория"] = pd.qcut(
    data['Стоимость (рубли)'], q=quantil, labels=price_category_labels
)

selected_features = ["общая (м²)", "Расстояние до метро (мин)", "Тип квартиры", "отделка"]
X = data[selected_features]
y = data["Ценовая категория"].values

categorical_features = ["Тип квартиры", "отделка"]
X = pd.get_dummies(X, columns=categorical_features).values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

max_depths = range(1, 11)
n_estimators_range = [10, 50, 100, 200]
tree_heights = []
train_errors_rf = []
test_errors_rf = []
train_errors_ada = []
test_errors_ada = []

for max_depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X_train, y_train)
    tree_heights.append(tree.get_depth())

    rf = RandomForestClassifier(n_estimators=50, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    train_errors_rf.append(1 - rf.score(X_train, y_train))
    test_errors_rf.append(1 - rf.score(X_test, y_test))

for n_estimators in n_estimators_range:
    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    ada.fit(X_train, y_train)
    train_errors_ada.append(1 - ada.score(X_train, y_train))
    test_errors_ada.append(1 - ada.score(X_test, y_test))

# Построение графиков
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(max_depths, tree_heights, label="Высота дерева", marker="o")
plt.xlabel("Максимальная глубина")
plt.ylabel("Высота дерева")
plt.title("Зависимость высоты дерева от максимальной глубины")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(max_depths, test_errors_rf, label="Ошибка на тесте (Random Forest)", marker="o")
plt.plot(max_depths, train_errors_rf, label="Ошибка на обучении (Random Forest)", marker="o")
plt.xlabel("Максимальная глубина")
plt.ylabel("Ошибка")
plt.title("Зависимость ошибки от максимальной глубины для Random Forest")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, test_errors_ada, label="Ошибка на тесте (AdaBoost)", marker="o")
plt.plot(n_estimators_range, train_errors_ada, label="Ошибка на обучении (AdaBoost)", marker="o")
plt.xlabel("Количество деревьев")
plt.ylabel("Ошибка")
plt.title("Зависимость ошибки от количества деревьев для AdaBoost")
plt.legend()
plt.show()
