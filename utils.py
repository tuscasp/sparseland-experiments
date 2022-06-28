import numpy as np

def squared_error(estimation: np.ndarray, ground_truth: np.ndarray):
    return np.sum((estimation - ground_truth)**2)

def normalize_vector_L2(v: np.ndarray):
    return v / np.sum(v**2)

def normalize_columns_L2(M: np.ndarray):
    norms = np.sqrt(np.sum(M * M, axis=0))
    norms_repeated = np.tile(norms, (M.shape[0],1))
    return M / norms_repeated, norms

def compute_psnr(v_true, v_estimated):
    dynamic_range = np.abs(np.max(v_true) - np.min(v_true))
    mse_val = np.sum((v_true - v_estimated)**2) / v_true.shape[0]**2
    psnr = 10 * np.log10(dynamic_range**2 / mse_val)
    return psnr
