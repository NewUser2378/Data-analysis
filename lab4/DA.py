import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None

    def fit(self, X, y, sample_weights=None):
        self.tree = self._build_tree(X, y, depth=0, sample_weights=sample_weights)

    def _gini(self, y, sample_weights=None):
        if sample_weights is None:
            sample_weights = np.ones(len(y)) / len(y)

        classes, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return 1 - np.sum(prob ** 2)

    def _split(self, X, y, feature_idx, threshold, sample_weights=None):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        left_weights = sample_weights[left_mask] if sample_weights is not None else None
        right_weights = sample_weights[right_mask] if sample_weights is not None else None

        return X[left_mask], X[right_mask], y[left_mask], y[right_mask], left_weights, right_weights

    def _find_best_split(self, X, y, sample_weights=None):
        best_feature, best_threshold = None, None
        best_impurity = self._gini(y, sample_weights)
        best_impurity_decrease = 0

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right, left_weights, right_weights = self._split(X, y, feature_idx,
                                                                                            threshold, sample_weights)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                impurity_left = self._gini(y_left, left_weights)
                impurity_right = self._gini(y_right, right_weights)
                impurity = (len(y_left) * impurity_left + len(y_right) * impurity_right) / len(y)
                impurity_decrease = best_impurity - impurity

                if impurity_decrease > best_impurity_decrease:
                    best_impurity_decrease = impurity_decrease
                    best_feature = feature_idx
                    best_threshold = threshold

        if best_threshold is None:
            return None, None, 0

        return best_feature, best_threshold, best_impurity_decrease

    def _build_tree(self, X, y, depth, sample_weights=None):
        num_samples, num_features = X.shape
        if (self.max_depth is not None and depth >= self.max_depth) or \
                (num_samples < self.min_samples_split) or \
                (self._gini(y, sample_weights) == 0):
            return {"type": "leaf", "class": np.argmax(np.bincount(y)), "depth": depth}

        best_feature, best_threshold, impurity_decrease = self._find_best_split(X, y, sample_weights)

        if impurity_decrease == 0:
            return {"type": "leaf", "class": np.argmax(np.bincount(y)), "depth": depth}

        if best_threshold is None:
            return {"type": "leaf", "class": np.argmax(np.bincount(y)), "depth": depth}

        X_left, X_right, y_left, y_right, left_weights, right_weights = self._split(X, y, best_feature, best_threshold,
                                                                                    sample_weights)

        return {
            "type": "node",
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X_left, y_left, depth + 1, sample_weights=left_weights),
            "right": self._build_tree(X_right, y_right, depth + 1, sample_weights=right_weights),
            "depth": depth
        }

    def _predict_one(self, x, tree):
        if tree["type"] == "leaf":
            return tree["class"]
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_one(x, tree["left"])
        else:
            return self._predict_one(x, tree["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def get_tree_height(self):
        def _get_height(node):
            if node["type"] == "leaf":
                return node["depth"]
            return max(_get_height(node["left"]), _get_height(node["right"]))

        return _get_height(self.tree)


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
            X_bootstrap, y_bootstrap = X[indices], y[indices]
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(pred).argmax() for pred in predictions.T])


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        sample_weights = np.ones(len(y)) / len(y)

        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=1, min_samples_split=2)


            tree.fit(X, y, sample_weights=sample_weights)
            predictions = tree.predict(X)

            incorrect = (predictions != y)
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

            alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))
            sample_weights = sample_weights * np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)

            self.alphas.append(alpha)
            self.models.append(tree)

    def predict(self, X):
        model_predictions = np.array([model.predict(X) for model in self.models])

        weighted_predictions = np.dot(self.alphas, model_predictions)

        return np.sign(weighted_predictions)


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


y = pd.factorize(y)[0]


from sklearn.model_selection import train_test_split
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
    tree = DecisionTree(max_depth=max_depth)
    tree.fit(X_train, y_train)
    tree_heights.append(tree.get_tree_height())

    rf = RandomForest(n_estimators=50, max_depth=max_depth)
    rf.fit(X_train, y_train)
    train_errors_rf.append(1 - np.mean(rf.predict(X_train) == y_train))
    test_errors_rf.append(1 - np.mean(rf.predict(X_test) == y_test))

for n_estimators in n_estimators_range:
    ada = AdaBoost(n_estimators=n_estimators)
    ada.fit(X_train, y_train)
    train_errors_ada.append(1 - np.mean(ada.predict(X_train) == y_train))
    test_errors_ada.append(1 - np.mean(ada.predict(X_test) == y_test))


plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(max_depths, tree_heights, label="Высота дерева", marker="o")
plt.xlabel("Максимальная глубина")
plt.ylabel("Высота дерева")
plt.title("Зависимость высоты дерева от max_depth")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(max_depths, test_errors_rf, label="Ошибка на тестовой выборке (Random Forest)", marker="o")
plt.plot(max_depths, train_errors_rf, label="Ошибка на обучающей выборке (Random Forest)", marker="o")
plt.xlabel("Максимальная глубина")
plt.ylabel("Ошибка")
plt.title("Зависимость ошибки от максимальной глубины для Random Forest")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, test_errors_ada, label="Ошибка на тестовой выборке (AdaBoost)", marker="o")
plt.plot(n_estimators_range, train_errors_ada, label="Ошибка на обучающей выборке (AdaBoost)", marker="o")
plt.xlabel("Количество деревьев")
plt.ylabel("Ошибка")
plt.title("Зависимость ошибки от количества деревьев для AdaBoost")
plt.legend()
plt.show()

