import numpy as np
from scipy import stats
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif

class EnhancedCustomKNN:
    def __init__(self, k_range=(3, 7), weights='adaptive'):
        self.k_range = k_range
        self.weights = weights
        self.optimal_k = None
        self.feature_weights = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self._compute_feature_weights(X, y)
        self._optimize_k(X, y)

    def _optimize_k(self, X, y):
        best_score = 0
        best_k = self.k_range[0]
        for k in range(self.k_range[0], self.k_range[1] + 1):
            score = self._cross_validate(X, y, k)
            if score > best_score:
                best_score = score
                best_k = k
        self.optimal_k = best_k

    def _cross_validate(self, X, y, k):
        n_splits = 5
        scores = []
        for i in range(n_splits):
            mask = np.zeros(len(X), dtype=bool)
            mask[np.random.choice(len(X), len(X) // n_splits, replace=False)] = True
            X_val, y_val = X[mask], y[mask]
            X_train, y_train = X[~mask], y[~mask]
            y_pred = self._predict(X_val, X_train, y_train, k)
            scores.append(np.mean(y_pred == y_val))
        return np.mean(scores)

    def _compute_feature_weights(self, X, y):
        self.feature_weights = mutual_info_classif(X, y)
        self.feature_weights = np.nan_to_num(self.feature_weights)
        self.feature_weights[self.feature_weights == None] = 0
        self.feature_weights /= np.sum(self.feature_weights)

    def predict(self, X):
        return self._predict(X, self.X_train, self.y_train, self.optimal_k)

    def _predict(self, X, X_train, y_train, k):
        y_pred = []
        for x in X:
            distances = np.sqrt(np.sum(((X_train - x) * self.feature_weights) ** 2, axis=1))
            k_indices = np.argsort(distances)[:k]
            k_nearest_labels = y_train.iloc[k_indices]

            if self.weights == 'uniform':
                y_pred.append(stats.mode(k_nearest_labels, keepdims=False).mode)
            elif self.weights == 'distance':
                k_distances = distances[k_indices]
                weights = 1 / (k_distances + 1e-8)
                weighted_votes = np.bincount(k_nearest_labels, weights=weights)
                y_pred.append(np.argmax(weighted_votes))
            elif self.weights == 'adaptive':
                k_distances = distances[k_indices]
                weights = np.exp(-k_distances)
                weighted_votes = np.bincount(k_nearest_labels, weights=weights)
                y_pred.append(np.argmax(weighted_votes))
        return np.array(y_pred)

    def score(self, X_test, y_test):
        return np.mean(self.predict(X_test) == y_test)

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        "Support Vector Machine": SVC(random_state=42),
        "Scikit-learn KNN": KNeighborsClassifier(n_neighbors=3, weights='uniform'),
        "Enhanced Custom KNN": EnhancedCustomKNN(k_range=(3, 7), weights='adaptive'),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    
    accuracy_results = {}
    confusion_matrices = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        if model_name == "Enhanced Custom KNN":
            y_pred = model.predict(X_test)
            accuracy = model.score(X_test, y_test)
        else:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[model_name] = accuracy
        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)
    
    return accuracy_results, confusion_matrices, models

