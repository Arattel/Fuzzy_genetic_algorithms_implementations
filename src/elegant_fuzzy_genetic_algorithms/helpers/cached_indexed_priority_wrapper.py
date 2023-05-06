import os 
from typing import Union
import pickle as pkl

import numpy as np
import faiss


from .parallel_priority_wrapper import ParallelPriorityWrapper
from ...common.approximation_helpers import (generate_search_space, init_param_index, estimate_by_index)

BASE_PATH: str = './indices/'


def calculate_priorities(params_combinations, n_terms_priority, nmax: int = 1000, membership_function='trapezoid'):
    """Calculates priorities given parameter combinations

    Args:
        params_combinations (_type_): _description_
        n_terms_params (_type_): _description_
        n_terms_priority (_type_): _description_
        nmax (int, optional): Max number of entries to be calculated at once. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    n_iterations: int = (params_combinations.shape[0] // nmax) + 1
    priority_inferencer = ParallelPriorityWrapper(n_terms_fitness=n_terms_priority, membership_function=membership_function)
    
    priorities_final = []
    for i in range(n_iterations):
        priorities_ = priority_inferencer.infer_priority(c1=params_combinations[i*nmax:(i + 1)* nmax, 0], 
                                                         c2=params_combinations[i*nmax:(i + 1)* nmax, 1])
        priorities_final.append(priorities_)
        
    priorities_final  = np.concatenate(priorities_final)
    return priorities_final


def priority_index_and_y(n_terms_fitness: int = 3, membership_fn: str = 'trapezoid') -> str:
    if membership_fn == 'trapezoid': 
        name = f'priority_EFGA_index_{n_terms_fitness}.index'
        name_y = f'priority_EFGA_y_{n_terms_fitness}.pkl'
    elif membership_fn == 'bell':
        name = f'priority_EFGA_index_{n_terms_fitness}_{membership_fn}_membership.index'
        name_y = f'priority_EFGA_y_{n_terms_fitness}_{membership_fn}_membership..pkl'
    return os.path.join(BASE_PATH, name), os.path.join(BASE_PATH, name_y)


class CachedPriorityWrapper:
    N_POINTS: Union[int, tuple[int]] = 100

    def __init__(self, n_terms_fitness: int = 3, membership_function='trapezoid') -> None:
        self.membership_function = membership_function
        self.n_terms = n_terms_fitness
        self.index_pth, self.y_pth = priority_index_and_y(n_terms_fitness, membership_fn=self.membership_function)

        # If we have index, we read it. Otherwise, we generate it and cache
        if os.path.exists(self.index_pth) and os.path.exists(self.y_pth):
            self.index = faiss.read_index(self.index_pth)

            with open(self.y_pth, 'rb') as f:
                self.y = pkl.load(f) 
            
        else:
            self._generate_index()
    
    def _generate_index(self):
        params_combinations = generate_search_space(self.N_POINTS)
        param_index = init_param_index(params_combinations)
        y =  calculate_priorities(params_combinations, self.n_terms, membership_function=self.membership_function)

        self.index = param_index
        self.y = y

        print(os.getcwd())
        faiss.write_index(self.index, self.index_pth)

        with open(self.y_pth, 'wb') as f:
            pkl.dump(self.y, f)



    def infer_priority(self, c1: np.array, c2: np.array) -> np.array:
        query = np.zeros(shape=(c1.shape[0], 2))
        query[:, 0] = c1
        query[:, 1] = c2
        return estimate_by_index(self.index, self.y, query=query)
