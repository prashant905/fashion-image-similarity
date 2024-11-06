import numpy as np
from sklearn.preprocessing import normalize
from datasketch import MinHash

# Utility function to create MinHash
def create_minhash(features, num_permutations):
    if features is None:
        return None

    binary_features = (features > 0).astype(int)
    minhash = MinHash(num_perm=num_permutations)
    for idx, val in enumerate(binary_features):
        if val == 1:
            minhash.update(str(idx).encode('utf-8'))
    return minhash