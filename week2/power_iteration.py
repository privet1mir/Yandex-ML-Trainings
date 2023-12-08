import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    r_0 = np.random.rand(data.shape[1])
    for _ in range(num_steps): 
        r = (data@r_0)/np.linalg.norm(data@r_0)
        r_0 = r
    mu = (r.T@data@r)/(r.T@r)
    return float(mu), r