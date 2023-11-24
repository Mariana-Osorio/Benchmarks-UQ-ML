import numpy as np

# Y = sobol-g(X) returns the value of the g-function or Sobol function.

# It is a commonly used benchmark for global sensitivity analysis with well-known analytical solutions for the Sobol' sensitivity indices.


def model(X, P):
    X = np.array(X,ndmin=2)

    assert P['dim'] == X.shape[1], "Parameter dimension and actual input dimension don't match."

    P = np.array(P['P'],ndmin=1).ravel()

    if len(P) == 1:
        P = np.ones((X.shape[1]))*P[0]
        
    assert X.shape[1] == len(P), f"Wrong input shapes. X.shape[1] = {X.shape[1]} must be same size as len(P) = {len(P)}"

    result = np.prod((np.abs(4 * X - 2) + P) / (1 + P), axis=1, keepdims=True)

    return result
