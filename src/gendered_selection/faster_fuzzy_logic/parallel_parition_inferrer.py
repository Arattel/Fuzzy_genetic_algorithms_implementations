import multiprocessing as mp
import numpy as np

from .generalized_partition_inferrer import GeneralizedInferrer
from ...common.parallel_utils import _extract_results



def infer_partner_ages(fs: GeneralizedInferrer, ages: np.array, diversity: float, var: mp.Queue, i: int):
    """Infers partner ages for given partition and puts it into 

    Args:
        fs (_type_): Generalized inferrer object
        ages (_type_): array of male ages
        diversity (_type_): diversity of population
        var (_type_): multiprocessing queue
        i (_type_): index of the process
    """
    var.put((np.array([fs.infer_partner_age(age=a, diversity=diversity) for a in ages]), i))

class ParallelInferrer:
    def __init__(self, n_partitions: int = 4, n_processes: int = mp.cpu_count(), membership_function='trapezoid') -> None:
        # Initilizing a separate inferrer for each process
        self.inferrers = [GeneralizedInferrer(n_partitions=n_partitions, membership_function=membership_function) for _ in range(n_processes)]
        self.n_processes = n_processes

    def preferred_age(self, male_indices_to_reproduce: np.array, lifetime: np.array, population_diversity: float) -> np.array:
        splits = np.array_split(male_indices_to_reproduce, self.n_processes)
        ctx = mp.get_context('fork')
        processes = [i for i in range(self.n_processes)]
        
        q = mp.Queue()
        for i, split in enumerate(splits):
            processes[i] = ctx.Process(target=infer_partner_ages, args=(self.inferrers[i], lifetime[split], population_diversity, q, i))
            processes[i].start()
            
        for i  in range(len(processes)):
            processes[i].join()
            
        return _extract_results(q)