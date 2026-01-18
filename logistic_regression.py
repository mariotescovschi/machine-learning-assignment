import numpy as np


class LogisticRegressionScratch:
    """Logistic Regression using Gradient Descent with optional L2 regularization."""

    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.0):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.reg = regularization
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, y, y_pred):
        m = len(y)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        cost = -1 / m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        if self.reg > 0:
            cost += (self.reg / (2 * m)) * np.sum(self.weights**2)
        return cost

    def fit(self, X, y, verbose=True):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.cost_history = []

        for i in range(self.n_iter):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            error = y_pred - y
            dw = (1 / m) * np.dot(X.T, error)
            db = (1 / m) * np.sum(error)

            if self.reg > 0:
                dw += (self.reg / m) * self.weights

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)

            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.n_iter}, Cost: {cost:.6f}")

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])


def precision_recall_f1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    return precision, recall, f1


def roc_auc(y_true, y_proba):
    sorted_indices = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    TP, FP = 0, 0
    tpr_list, fpr_list = [0], [0]

    for label in y_true_sorted:
        if label == 1:
            TP += 1
        else:
            FP += 1
        tpr_list.append(TP / P if P > 0 else 0)
        fpr_list.append(FP / N if N > 0 else 0)

    auc = 0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2
    return auc
