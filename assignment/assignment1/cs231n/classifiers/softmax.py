from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange



def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops).

    Inputs:
    - W: Weights of shape (D, C)
    - X: Data of shape (N, D)
    - y: Labels of shape (N,)
    - reg: Regularization strength

    Returns:
    - loss: Scalar loss
    - dW: Gradient of W
    """
    # Initialize loss and gradient
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        # Compute class scores
        scores = X[i].dot(W)

        # Numerical stability
        scores -= np.max(scores)

        # Compute softmax probabilities
        p = np.exp(scores)
        p /= np.sum(p)

        # Compute loss
        loss -= np.log(p[y[i]])

        # Compute gradient
        ds = p.copy()
        ds[y[i]] -= 1.0

        for j in range(num_classes):
            dW[:, j] += X[i] * ds[j]

    # Average loss and gradient
    loss /= num_train
    dW /= num_train

    # Add L2 regularization
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW



def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    """
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    
    # 1. 计算得分
    scores = X.dot(W)
    
    # 2. 数值稳定
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # 3. 计算指数和概率
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # 4. 计算交叉熵损失
    correct_logprobs = -np.log(probs[np.arange(num_train), y])
    data_loss = np.sum(correct_logprobs) / num_train
    
    # 5. 加上正则化项 (使用 0.5 的系数)
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss
    
    # ================= 下面补充梯度计算 =================
    # 6. 计算 dscores
    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1
    dscores /= num_train
    
    # 7. 计算 dW
    dW = X.T.dot(dscores)
    
    # 8. 加上正则化梯度 (与 0.5 系数对应，求导后直接是 reg * W)
    dW += reg * W
    
    return loss, dW
