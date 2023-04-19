from typing import Union


import numpy as np
import faiss


def generate_search_space(n_splits: Union[tuple[int], int], ranges: list[tuple[float]]  = [(0, 1), (0, 1)]):
    """Generates space of all combinations of points in range

    Args:
        n_splits (Union[tuple[int], int]): Number of points for each range
        ranges (list[tuple[float]], optional): Ranges. Defaults to [(0, 1), (0, 1)].

    Returns:
        _type_: Search space
    """

    # Making it tuple
    if isinstance(n_splits, int):
        n_splits = (n_splits, n_splits)

    
    range_1 = np.linspace(start=ranges[0][0], stop=ranges[0][1], num=n_splits[0])
    range_2 = np.linspace(start=ranges[1][0], stop=ranges[1][1], num=n_splits[1])

    params_combinations = np.array(np.meshgrid(range_1, range_2)).T.reshape(-1, 2)

    return params_combinations



def init_param_index(params_combinations: np.array) -> faiss.Index:
    """Given combination of parameters creates Index object for faster search

    Args:
        params_combinations (np.array):  Search space

    Returns:
        faiss.Index: Faiss index object
    """
    param_index = faiss.IndexFlatL2(params_combinations.shape[1])
    param_index.add(params_combinations)
    return param_index


def estimate_by_index(index: faiss.Index, y: np.array, query: np.array):
    D, I = index.search(query, k=1)
    estimations = y[I]
    return estimations