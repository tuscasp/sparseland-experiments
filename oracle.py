import numpy as np

def oracle(A: np.ndarray, b: np.ndarray, s: np.ndarray):
    '''
    ORACLE estimator, solving the following problem:
    min_x ||b - Ax||_2^2 s.t. supp{x} = s

    Parameters:
    * A: n x m matrix, with m >= n
    * b: vector or column matrix of size n
    * s: support (index of dense entries) of the solution

    Return:
    * x: solution for linear system
    '''

    # Initialize the vector x
    x = np.zeros(np.shape(A)[1])

    A_support = A[:, s]
    x[s] = np.linalg.pinv(A_support.T @ A_support) @ A_support.T @ b

    return x