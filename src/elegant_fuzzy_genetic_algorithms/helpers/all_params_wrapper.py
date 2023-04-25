from src.elegant_fuzzy_genetic_algorithms.helpers.generalized_param_inferrer import GeneralizedParamInferencer
from src.elegant_fuzzy_genetic_algorithms.helpers.cached_indexed_priority_wrapper import CachedPriorityWrapper
from src.elegant_fuzzy_genetic_algorithms.helpers.parallel_priority_wrapper import ParallelPriorityWrapper
from .param_inference_approximation import ParamInferenceApprox

import multiprocessing as mp
import numpy as np


class AllEFGAParamsParallelWrapper:
    def __init__(self, n_terms_params: int = 3, n_terms_priority: int = 3, n_processes: int = mp.cpu_count(), 
                 use_approx=True) -> None:
        self.priority = CachedPriorityWrapper(n_terms_fitness=n_terms_priority) if use_approx else ParallelPriorityWrapper(n_terms_fitness=n_terms_priority, 
                                                                                                                           n_processes=n_processes)
        self.params = GeneralizedParamInferencer(n_terms_params) if not use_approx else ParamInferenceApprox()
    
    def infer_priority(self, c1, c2):
        return self.priority.infer_priority(c1, c2)
    
    def infer(self, bestFitness: float, avgFitness: float, avgFitChange: float) -> float:
        return self.params.infer(bestFitness=bestFitness, avgFitness=avgFitness, avgFitChange=avgFitChange)