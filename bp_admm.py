import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve_triangular
import pywt

def bp_admm(A: np.ndarray, b: np.ndarray, lmbda: float, max_iters: int = 200, tol_admm = 1e-4):
    '''
    BP_ADMM Solve Basis Pursuit problem via ADMM

    Solves the following problem:
    min_x 1/2*||b - CAx||_2^2 + lmbda*|| x ||_1

    Parameters:
    * A: n x m matrix, with m >= n
    * b: vector or column matrix of size n
    * lmbda: regularization parameter of loss function
    * max_ters: max #iterations to perform
    * tol_admm: tolerance of convergence for admm iteration.

    Return:
    * x: solution for linear system
    '''

    # Use cholesky factorization for efficiently solving linear system Ax=b
    n, m = A.shape
    L = np.linalg.cholesky(A.T @ A + np.eye(m))
    U = L.T

    v = np.zeros(m, float)
    v_prev = np.zeros_like(v)
    u = np.zeros_like(v)

    done = False
    iter = 0
    signal_atoms_alignment = A.T @ b

    while not done:

        # solve linear system
        # (A'*A + I)x = (A^t b + v - u)
        y = solve_triangular(L, signal_atoms_alignment + v - u, lower=True)
        x = solve_triangular(U, y, lower=False)

        v = pywt.threshold(x+u, lmbda,'soft')
        u = u + x - v

        iter += 1
        done_by_iterations = max_iters < iter
        if np.linalg.norm(v):
            done_by_convergence = (norm(v - v_prev) / norm(v + 1e-12)) < tol_admm
        else:
            done_by_convergence = False
        done = done_by_iterations or done_by_convergence

        if done_by_iterations:
            print('Exiting, but BP-ADMM did not converge')

        v_prev = v

    return v

