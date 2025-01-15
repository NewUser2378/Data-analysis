import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt

df = pd.read_csv('SMS.tsv', sep='\t').head(500)

y = df['class'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

class CustomKMeans:
    def __init__(self, n_clusters=2, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def cluster_centers_(self):
        return self.centroids

def custom_filter_method(X, y, k=30):
    f_scores, _ = f_classif(X, y)
    selected_indices = np.argsort(f_scores)[-k:]
    return X[:, selected_indices], selected_indices

def custom_wrapper_method(X, y, k=30):
    model = LogisticRegression(max_iter=1000)
    sfs = SFS(estimator=model,
              k_features=k,
              forward=True,
              floating=False,
              scoring='accuracy',
              cv=5)
    sfs = sfs.fit(X, y)
    selected_indices = sfs.k_feature_idx_
    X_selected = X[:, selected_indices]
    return X_selected, selected_indices

def custom_embedded_method(X, y, k=30):
    model = LassoCV(cv=5).fit(X, y)
    importance = np.abs(model.coef_)
    selected_indices = np.argsort(importance)[-k:]
    return X[:, selected_indices], selected_indices

def evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

print("Качество классификаторов до выбора признаков:")
for name, model in models.items():
    accuracy = evaluate_model(X_train, X_test, y_train, y_test, model)
    print(f"{name}: {accuracy:.4f}")

custom_methods = {
    'Custom Filter': custom_filter_method(X.toarray(), y, k=30),
    'Custom Wrapper': custom_wrapper_method(X.toarray(), y, k=30),
    'Custom Embedded': custom_embedded_method(X.toarray(), y, k=30)
}

print("\nКачество классификаторов после выбора признаков (кастомные методы):")
for method_name, (X_selected, _) in custom_methods.items():
    print(f"\nМетод выбора признаков: {method_name}")
    for name, model in models.items():
        accuracy = evaluate_model(X_train[:, :30], X_test[:, :30], y_train, y_test, model)
        print(f"{name}: {accuracy:.4f}")

    # Выводим 30 предсказанных слов для каждого метода
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_30_words = feature_names[_[::-1]]  # Индексы отсортированы по убыванию значимости
    print(f"Топ 30 предсказанных слов для метода {method_name}:")
    print(top_30_words[:30])
    print("\n")

def evaluate_clustering(true_labels, predicted_labels, X):
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    silhouette_avg = silhouette_score(X, predicted_labels)

    print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    return ari_score, silhouette_avg

def visualize_clusters(X, true_labels, predicted_labels, method_name="PCA"):
    if method_name == "PCA":
        pca = PCA(n_components=2)
        reduced_X = pca.fit_transform(X)
    elif method_name == "t-SNE":
        tsne = TSNE(n_components=2, random_state=42)
        reduced_X = tsne.fit_transform(X)

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_X[:, 0], reduced_X[:, 1], c=true_labels, cmap='viridis', label='True labels', alpha=0.5)
    plt.scatter(reduced_X[:, 0], reduced_X[:, 1], c=predicted_labels, cmap='coolwarm', marker='x', label='Cluster labels')
    plt.title(f"Cluster Visualization using {method_name}")
    plt.legend()
    plt.show()

def kmeans_clustering(X, n_clusters=2):
    kmeans = CustomKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels, kmeans.cluster_centers_

labels_before, centers_before = kmeans_clustering(X.toarray())
ari_before, silhouette_before = evaluate_clustering(y, labels_before, X.toarray())

visualize_clusters(X.toarray(), y, labels_before, method_name="PCA")
visualize_clusters(X.toarray(), y, labels_before, method_name="t-SNE")

labels_after, centers_after = kmeans_clustering(custom_methods['Custom Filter'][0])
ari_after, silhouette_after = evaluate_clustering(y, labels_after, custom_methods['Custom Filter'][0])

visualize_clusters(custom_methods['Custom Filter'][0], y, labels_after, method_name="PCA")
visualize_clusters(custom_methods['Custom Filter'][0], y, labels_after, method_name="t-SNE")

print("\nОценка качества кластеризации до выбора признаков:")
print(f"Adjusted Rand Index (ARI): {ari_before:.4f}")
print(f"Silhouette Score: {silhouette_before:.4f}")

print("\nОценка качества кластеризации после выбора признаков (фильтрация):")
print(f"Adjusted Rand Index (ARI): {ari_after:.4f}")
print(f"Silhouette Score: {silhouette_after:.4f}")
