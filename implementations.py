import numpy as np

#Berta ( he copitat les funcions que tenia i ho he fet amb funcions auxiliars poer si es millor sense funcions auxiliars ho canvioo, i  obviament podeu canviar el que volgueu!!) : 

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE."""
    e = y - tx @ w 
    (N,_) = tx.shape
    L = ( 1/(2*N) )*np.sum(e**2)
    return L
    
def compute_gradient(y, tx, w):
    """Computes the gradient at w. """
    e = y  - tx @ w
    N = len(y)
    gradient = -(1 / N) * (tx.T @ e)
    return gradient

def mean_squared_error_gd(y, x, initial_w, max_iters, gamma,tol = 1e-8):
    """Linear regression using gradient descent."""
    w = np.copy(initial_w)
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    losses = []
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
        loss = compute_loss(y, tx, w)
        losses.append(loss)

        if n_iter > 0 and abs(losses[-1] - losses[-2]) < tol:
                print(f"Converged at iteration {n_iter}")
                break
       
        if n_iter % 100 == 0:
            print(f"Iter {n_iter:5d} | Loss = {loss:.6f} | Grad = {np.linalg.norm(gradient):.4f}")

    loss = compute_loss(y, tx, w)
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size = 1)."""
    w = np.copy(initial_w)
    num_samples = len(y)

    for n_iter in range(max_iters):
        n = np.random.randint(0, num_samples)
        y_batch = y[n:n+1]
        tx_batch = tx[n:n+1]

        gradient_stoch = compute_gradient(y_batch, tx_batch, w)
        w = w - gamma * gradient_stoch
    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares(y, x):
    """Least squares regression using normal equations."""
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    A = tx.T @ tx
    b = tx.T @ y
    
    w = np.linalg.solve(A,b)
    
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    N = len(y)
    D = tx.shape[1]
    
    A = tx.T @ tx + (2 * N * lambda_) * np.eye(D)
    b = tx.T @ y
    
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)
    
    return w, loss

def sigmoid(t):
    return 1/(1 + np.exp(-t)) 
    
def calculate_loss_logistic(y,tx,w): 
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    N = len(y)
    eps = 1e-15
    pred = sigmoid(tx @ w)
    pred = np.clip(pred, eps, 1.0 - eps)
    L = (-1 / N) * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return float(L)

def calculate_gradient_logistic(y,tx,w): 
    N = len(y)
    pred = sigmoid(tx @ w)
    grad = (1 / N) * tx.T @ (pred - y) 
    return grad
     

def logistic_regression(y, x, initial_w, max_iters, gamma,tol=1e-8):
    """Logistic regression using gradient descent."""
    w = np.copy(initial_w)
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    losses = []
    
    for n_iter in range(max_iters):
        grad = calculate_gradient_logistic(y,tx,w)
        w = w - gamma*grad
        loss = calculate_loss_logistic(y, tx, w)
        losses.append(loss)
        if n_iter > 0 and abs(losses[-1] - losses[-2]) < tol:
            print(f"Converged at iteration {n_iter}")
            break
   
        if n_iter % 100 == 0:
            print(f"Iter {n_iter:5d} | Loss = {loss:.6f} | Grad = {np.linalg.norm(grad):.4f}")

    loss = calculate_loss_logistic(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent."""
    w = np.copy(initial_w)
    N = len(y)
    
    for n_iter in range(max_iters):
        pred = sigmoid(tx @ w)
        gradient = (1 / N) * tx.T @ (pred - y)  + 2*lambda_*w
        w = w - gamma*gradient
        
    log_loss = calculate_loss_logistic(y, tx, w)
    loss = log_loss + lambda_ * np.sum(w**2) 
    return w, loss
    