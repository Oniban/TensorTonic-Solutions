import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
        # Case 1: empty input
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=int)
    
    # Case 2: determine max_len
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    # Initialize result array with pad_value
    result = np.full((len(seqs), max_len), pad_value, dtype=int)
    
    for i, seq in enumerate(seqs):
        # Truncate if longer
        truncated = seq[:max_len]
        
        # Fill into result
        result[i, :len(truncated)] = truncated
    
    return result