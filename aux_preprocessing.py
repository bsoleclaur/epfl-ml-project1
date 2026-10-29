import numpy as np
import matplotlib.pyplot as plt
from aux_cleaning_functions import *
from aux_exploratory_data_analysis import detect_categorical_features
from implementations import *


def sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def weighted_logistic_loss(y, tx, w, pos_weight=1.0, neg_weight=1.0):
    """
    Weighted (and optionally penalized) logistic loss.
    """
    p = sigmoid(tx @ w)
    eps = 1e-15  # avoid log(0)

    # weights per sample
    sample_weights = np.where(y == 1, pos_weight, neg_weight)

    # weighted average loss
    loss = -np.sum(
        sample_weights * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    ) / np.sum(sample_weights)

    # no regularization term added to returned loss (for monitoring only)
    return float(loss)


def weighted_gradient_logistic(y, tx, w, lambda_=0.0, pos_weight=1.0, neg_weight=1.0):
    """
    Gradient of the weighted logistic loss with L2 penalty.
    """
    p = sigmoid(tx @ w)
    sample_weights = np.where(y == 1, pos_weight, neg_weight)
    error = sample_weights * (p - y)
    grad = (tx.T @ error) / np.sum(sample_weights)
    grad[1:] += 2 * lambda_ * w[1:]  # don't regularize bias
    return grad.ravel()


def logistic_regression_weighted_gd(
    y,
    x,
    lambda_=1e-3,
    gamma=0.05,
    pos_weight=1.0,
    neg_weight=1.0,
    max_iter=10000,
    tol=1e-4,
    clip_grad=10.0,
    verbose=True,
):
    """
    Logistic regression with class weights and L2 regularization.
    Returns (loss, w).
    """
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    w = np.zeros(tx.shape[1])
    losses = []

    for it in range(max_iter):
        grad = weighted_gradient_logistic(y, tx, w, lambda_, pos_weight, neg_weight)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > clip_grad:
            grad *= clip_grad / grad_norm

        loss = weighted_logistic_loss(y, tx, w, pos_weight, neg_weight)
        losses.append(loss)

        w -= gamma * grad

        if it > 0 and abs(losses[-1] - losses[-2]) < tol:
            if verbose:
                print(f"Converged at iteration {it}")
            break

        if verbose and it % 100 == 0:
            print(f"Iter {it:5d} | Loss = {loss:.6f} | GradNorm = {grad_norm:.4f}")

    return losses[-1], w


def predict_with_threshold(x, w, threshold=0.5):
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    probs = sigmoid(tx @ w)
    preds = (probs >= threshold).astype(int)
    return preds, probs


def predict_linear(x, w, threshold=0.5):

    tx = np.c_[np.ones((x.shape[0], 1)), x]
    predictions = tx @ w
    preds = (predictions >= threshold).astype(int)
    return preds, predictions


def f1_score(y_true, y_pred):
    """
    Compute F1 score using only NumPy.
    Works for binary classification (0/1).
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)
    f1 = 2 * precision * recall / (precision + recall + 1e-15)
    return f1


# --- Full evaluation wrapper ---
def evaluate_model(y_true, X, w, threshold=0.5):
    """
    Evaluate trained logistic regression on a dataset.
    Returns accuracy and F1 score.
    """
    preds, probs = predict_with_threshold(X, w, threshold=threshold)
    acc = accuracy_numpy(y_true, preds)
    f1 = f1_score(y_true, preds)
    print(f" Accuracy: {acc*100:.2f}%")
    print(f" F1 Score: {f1:.4f}")
    return acc, f1


def accuracy_numpy(y_true, y_pred):
    """
    Compute accuracy.
    """
    return np.mean(y_true == y_pred)


def recall_numpy(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall


def confusion_matrix_numpy(y_true, y_pred, method="Weighted", plot=True):
    """
    Compute and optionally plot the confusion matrix using NumPy.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted binary labels (0 or 1)
    plot : bool, optional
        If True, displays the confusion matrix as a heatmap

    Returns
    -------
    cm : np.ndarray
        2x2 confusion matrix [[TN, FP], [FN, TP]]
    """

    # Ensure binary
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    cm = np.array([[tn, fp], [fn, tp]])

    print("Confusion Matrix:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    if plot:
        plt.figure(figsize=(4, 3))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix {method}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.xticks([0, 1], ["0", "1"])
        plt.yticks([0, 1], ["0", "1"])

        for i in range(2):
            for j in range(2):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12
                )

        plt.tight_layout()
        plt.show()

    return cm


def impute_categorical_missing_code(X, cat_mask, missing_code=-1.0):
    """
    Replace NaNs in *categorical* columns with a special code (default -1),
    leaving numeric columns unchanged.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Full feature matrix (numeric + categorical).
    cat_mask : np.ndarray of bool, shape (n_features,)
        True for categorical columns, False for numeric ones.
    missing_code : float, default=-1.0
        Code used to represent 'Missing' category.

    Returns
    -------
    X_imputed : np.ndarray, same shape as X
        Copy of X where NaNs in categorical columns are replaced by `missing_code`.
    """
    Xf = np.array(X, dtype=np.float64, copy=True)
    cat_idx = np.where(cat_mask)[0]

    for j in cat_idx:
        col = Xf[:, j]
        nan_mask = np.isnan(col)
        if np.any(nan_mask):
            col[nan_mask] = missing_code
            Xf[:, j] = col

    return Xf
