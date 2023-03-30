import numpy as np
import multiprocessing as mp

def _extract_results(q: mp.Queue) -> np.array:
    """Extract results from multiprocessing queue

    Args:
        q (mp.Queue): multiprocessing queue

    Returns:
        np.array: female preferred age calculated
    """
    results = []
    
    # Simple dump of results to array
    while not q.empty():
        results.append(q.get())

    # Combining results into a single array with correct order
    results = sorted(results, key= lambda x: x[1])
    results = list(map(lambda x: x[0], results))
    return np.concatenate(results)
