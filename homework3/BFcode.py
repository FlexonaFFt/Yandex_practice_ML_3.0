class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance
        
import numpy as np


def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """  
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    matches = []
    
    # Compute pairwise distances between all descriptors
    distances = np.sqrt(np.sum((des1[:, np.newaxis] - des2) ** 2, axis=2))
    
    # For each descriptor in des1, find the best match in des2
    best_in_des2 = np.argmin(distances, axis=1)
    min_distances_des2 = np.min(distances, axis=1)
    
    # For each descriptor in des2, find the best match in des1
    best_in_des1 = np.argmin(distances, axis=0)
    min_distances_des1 = np.min(distances, axis=0)
    
    # Cross-check: keep only mutual best matches
    for i in range(des1.shape[0]):
        j = best_in_des2[i]
        if best_in_des1[j] == i:  # mutual best match
            matches.append(DummyMatch(i, j, min_distances_des2[i]))
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    return matches