import numpy as np
from numba import prange

def rewin(ts):
    calculate_self_similarity_matrix(ts)
    return 1


def calculate_self_similarity_matrix(ts):
    n = len(ts)
    sm = np.full((n, n), -np.inf, dtype=np.float32)
    # set gamma like in locomotif
    gamma = 1.0 / (np.std(ts, axis=None)**2)
    
    for i in prange(n):
        j_start =  i
        j_end = n
        
        # calculate similarities
        similarities = np.exp(-gamma * (ts[j_start:j_end] - ts[i])**2)

        sm[i, j_start:j_end] = similarities

    return sm