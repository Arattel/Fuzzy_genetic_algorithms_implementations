import multiprocessing as mp
import numpy as np

from .generalized_partition_inferrer import GeneralizedInferrer


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
    def __init__(self, n_partitions: int = 4, n_processes: int = mp.cpu_count()) -> None:
        # Initilizing a separate inferrer for each process
        self.inferrers = [GeneralizedInferrer(n_partitions=n_partitions) for _ in range(n_processes)]
        self.n_processes = n_processes

    def multiprocessing_preferred_age(self, male_indices_to_reproduce: np.array, lifetime: np.array, population_diversity: float) -> np.array:
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