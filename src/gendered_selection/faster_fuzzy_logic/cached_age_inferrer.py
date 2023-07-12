from typing import Union

import numpy as np
import faiss
import pickle as pkl
from tqdm import tqdm
import os


from src.common.approximation_helpers import (generate_search_space, init_param_index, estimate_by_index)
from src.gendered_selection.faster_fuzzy_logic.generalized_partition_inferrer import GeneralizedInferrer


BASE_PATH: str = './indices/'
def age_index_and_y(n_partitions: int = 5, membership_function='trapezoid') -> str:
    if membership_function == 'trapezoid':
        name = f'age_gendered_index_{n_partitions}.index'
        name_y = f'age_gendered_y_{n_partitions}.pkl'
    elif membership_function == 'bell':
        name = f'age_gendered_index_{n_partitions}_{membership_function}_membership.index'
        name_y = f'age_gendered_y_{n_partitions}_{membership_function}_membership.pkl'

    return os.path.join(BASE_PATH, name), os.path.join(BASE_PATH, name_y)

class CachedAgeEstimator:
    N_POINTS: Union[int, tuple[int]] = 200
    def __init__(self, n_partitions: int = 5, approx: bool = True, membership_function='trapezoid',  t_conorm=None, t_norm=None) -> None:
        self.n_partitions = n_partitions
        self.approx = approx
        self.membership_function = membership_function
        self.t_norm = t_norm
        self.t_conorm = t_conorm

        self.index_pth, self.y_pth = age_index_and_y(n_partitions, membership_function=membership_function)

        # If we have index, we read it. Otherwise, we generate it and cache
        if os.path.exists(self.index_pth) and os.path.exists(self.y_pth):
            self.index = faiss.read_index(self.index_pth)

            with open(self.y_pth, 'rb') as f:
                self.y = pkl.load(f) 
            
        else:
            self._generate_index()

    def _generate_index(self):
        # Creating an inferrer
        self.inferrer = GeneralizedInferrer(self.n_partitions, membership_function=self.membership_function,  t_conorm=self.t_conorm, 
                                            t_norm=self.t_norm)

        # Creating search space and index
        params_combinations = generate_search_space(n_splits=self.N_POINTS, ranges=[(0, 1), (0, 10)])
        param_index = init_param_index(params_combinations=params_combinations, approx_inf=self.approx)

        # Generating inference fast
        y  = np.array([self.inferrer.infer_partner_age(*params_combinations[i, :]) for i in tqdm(range(params_combinations.shape[0]))])


        self.index = param_index
        self.y = y

        print(os.getcwd())
        faiss.write_index(self.index, self.index_pth)

        with open(self.y_pth, 'wb') as f:
            pkl.dump(self.y, f)

        
    def preferred_age(self, male_indices_to_reproduce: np.array, lifetime: np.array, population_diversity: float) -> np.array:
        lifetimes_male = lifetime[male_indices_to_reproduce]
        population_diversity = np.repeat([population_diversity], lifetimes_male.shape[0])
        query = np.zeros(shape=(male_indices_to_reproduce.shape[0], 2))
        query[:, 0], query[:, 1]  = lifetimes_male, population_diversity
        return estimate_by_index(self.index, self.y, query=query).ravel()