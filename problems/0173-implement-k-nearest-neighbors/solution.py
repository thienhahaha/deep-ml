import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """

    points = np.array(points)
    query_point = np.array(query_point)
    Xbar = points.T
    d, N = Xbar.shape

    D = np.zeros((N, 1))
    for i in range(N):
        diff = Xbar[:, i] - query_point.T
        D[i, 0] = np.sum(diff * diff)

    indices = np.argsort(D, axis=0)[:k, 0]

    return [tuple(points[i]) for i in indices]

    pass