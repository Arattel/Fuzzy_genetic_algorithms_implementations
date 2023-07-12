import multiprocessing as mp
import numpy as np


from src.common.parallel_utils import _extract_results
from .generalized_priority_inferencer import GeneralizedPriorityInferencer

def infer_priority(fs: GeneralizedPriorityInferencer, c1, c2, var: mp.Queue, i: int):
    return var.put((np.array([fs.infer_priority(c1[i], c2[i]) for i in range(c1.shape[0])]), i))

class ParallelPriorityWrapper:
    def __init__(self, n_terms_fitness: int = 3, n_processes: int = mp.cpu_count(), membership_function='trapezoid', 
                 t_conorm=None, t_norm=None) -> None:
        self.inferrers = [GeneralizedPriorityInferencer(n_terms_fitness=n_terms_fitness, membership_function=membership_function, 
                                                        t_norm=t_norm, t_conorm=t_conorm) for _ in range(n_processes)]
        self.n_processes = n_processes

    def infer_priority(self, c1: np.array, c2: np.array) -> np.array:
        indices = np.arange(c1.shape[0])
        splits = np.array_split(indices, self.n_processes)
        ctx = mp.get_context('fork')
        processes = [i for i in range(self.n_processes)]
        
        q = mp.Queue()
        for i, split in enumerate(splits):
            processes[i] = ctx.Process(target=infer_priority, args=(self.inferrers[i], c1[split], c2[split], q, i))
            processes[i].start()
            
        for i  in range(len(processes)):
            processes[i].join()
            
        return _extract_results(q)