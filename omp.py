import numpy as np

def omp(A: np.ndarray, b: np.ndarray, max_support: int = -1):
    '''
    Implements ortoghonal matching pursuit (OMP) for solving linear system over noisy b given
    support size:
    || b - Ax ||_2 ^ 2 < eps ^ 2

    Parameters:
    * A: n x m matrix, with m >= n
    * b: vector or column matrix of size n
    * eps: acceptable error
    * max_support: maximal support size of vector x. If set to non-positive value, it is
    is considered to equal m.

    Return:
    * x: solution for linear system
    '''

    n, m = A.shape

    residual_vector = b
    residual_sqrd  = np.sum(residual_vector ** 2)
    support = set()

    if max_support <= 0:
        max_support = m

    done = False

    while not done :
        atoms_projection_into_residual = A.T @ residual_vector

        index_max_projection = np.argmax(np.abs(atoms_projection_into_residual))

        support.add(index_max_projection)

        atoms_chosen = A[:, list(support)]
        x_dense = np.linalg.inv(atoms_chosen.T @ atoms_chosen) @ atoms_chosen.T @ b

        residual_vector = b - atoms_chosen @ x_dense.T
        residual_sqrd  = np.sum(residual_vector ** 2)

        if len(support) >= max_support:
            done = True

    x = np.zeros(m)
    x[list(support)] = x_dense

    return x