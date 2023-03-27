import multiprocessing as mp

import numpy as np

from src.gendered_selection.conf.gendered_selection_config import Config
from src.gendered_selection.faster_fuzzy_logic.infer_partner_age import Inferrer


def infer_partner_ages(pth, ages, diversity, var, i):
    fs =  Inferrer(rule_path=pth)
    var.put((np.array([fs.infer_partner_age(age=a, diversity=diversity) for a in ages]), i))



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


def multiprocessing_preferred_age(male_indices_to_reproduce: np.array, lifetime: np.array, population_diversity: float, N_proc: int = mp.cpu_count(),) -> np.array:
    """Wrapper for parallel preferred age calculator

    Args:
        male_indices_to_reproduce (np.array): male indices selected for preproduction
        lifetime (np.array): array of lifetime for all genomes
        population_diversity (float): diversity metric
        N_proc (int, optional): Number of processes. Defaults to number of CPU cores. 

    Returns:
        np.array: calculated female preferred age
    """
    splits = np.array_split(male_indices_to_reproduce, N_proc)
    ctx = mp.get_context('fork')
    processes = [i for i in range(N_proc)]

    # Launching 
    q = mp.Queue()
    for i, split in enumerate(splits):
        processes[i] = ctx.Process(target=infer_partner_ages, args=(Config.RULES_FILE, lifetime[split], population_diversity, q, i))
        processes[i].start()
    
    for i  in range(len(processes)):
        processes[i].join()
    
    return _extract_results(q)