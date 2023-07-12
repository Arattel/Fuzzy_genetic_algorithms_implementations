import os 
from typing import Union
import pickle as pkl

import numpy as np
import faiss


from .parallel_priority_wrapper import ParallelPriorityWrapper
from ...common.approximation_helpers import (generate_search_space, init_param_index, estimate_by_index)
from ...common.naming import _base_file_name


BASE_PATH: str = './indices/'


def calculate_priorities(params_combinations, n_terms_priority, nmax: int = 1000, membership_function='trapezoid', t_conorm=None, 
                         t_norm=None):
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
    priority_inferencer = ParallelPriorityWrapper(n_terms_fitness=n_terms_priority, membership_function=membership_function, 
                                                  t_conorm=t_conorm, t_norm=t_norm)
    
    priorities_final = []
    for i in range(n_iterations):
        priorities_ = priority_inferencer.infer_priority(c1=params_combinations[i*nmax:(i + 1)* nmax, 0], 
                                                         c2=params_combinations[i*nmax:(i + 1)* nmax, 1])
        priorities_final.append(priorities_)
        
    priorities_final  = np.concatenate(priorities_final)
    return priorities_final


def priority_index_and_y(n_terms_fitness: int = 3, membership_fn: str = 'trapezoid', t_conorm: str = 'max', t_norm: str='min') -> str:
    name = _base_file_name('efga', n_terms_fitness, membership_fn, t_norm=t_norm, t_conorm=t_conorm)
    index_name = f'{name}.index'
    y_name = f'{name}_y.pkl'
    return os.path.join(BASE_PATH, index_name), os.path.join(BASE_PATH, y_name)


class CachedPriorityWrapper:
    N_POINTS: Union[int, tuple[int]] = 100

    def __init__(self, n_terms_fitness: int = 3, membership_function='trapezoid', 
                 t_conorm: str = 'max', t_norm: str='min') -> None:
        self.membership_function = membership_function
        self.n_terms = n_terms_fitness
        self.index_pth, self.y_pth = priority_index_and_y(n_terms_fitness, membership_fn=self.membership_function, 
                                                          t_conorm=t_conorm, t_norm=t_norm)
        self.t_norm = t_norm
        self.t_conorm = t_conorm

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
        y =  calculate_priorities(params_combinations, self.n_terms, membership_function=self.membership_function, 
                                  t_conorm=self.t_conorm, t_norm=self.t_conorm)

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
