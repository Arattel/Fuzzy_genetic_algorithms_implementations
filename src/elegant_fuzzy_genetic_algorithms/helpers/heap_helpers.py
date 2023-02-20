import heapq as hq
import numpy as np

def create_heap(fitness_arr: np.array) -> list[tuple]:
    """Function to generate min heap with priorities from fitness array (smaller fitness is better, so the most fit individual is on top of the heap)

    Args:
        fitness_arr (np.array): numpy array of float fitness values
    Returns:
        list[tuple]: heap in the list format [(priority, index in genome array, fitness)]
    """
    h = []
    for i in range(fitness_arr.shape[0]):
        hq.heappush(h, (fitness_arr[i], i, fitness_arr[i]))
    return h

def approximate_topk_indices(heap: list, k: int) -> list[int]:
    """Returns genome indices of top k best individuals (with the smallest fitness)

    Args:
        heap (list): heap in the form of list, [(priority, index in genome array, fitness)]
        k (int): number of indices of the best individuals to be returned

    Returns:
        list[int]: list of indices of the best individuals
    """
    topk = heap[:k]
    return list(map(lambda x: x[1], topk))


def update_parents_with_indices(heap: list[tuple], parent_indices: list[int], update_values: list[float]) -> list:
    """Updates parents' priorities in heap given indices in heap and values to be added to the priorities

    Args:
        heap (list[tuple]): heap in the format of list
        parent_indices (list[int]): heap list indices of parents to update
        update_values (list[float]): values to be added for each parent

    Returns:
        list: _description_
    """
    for i in range(len(parent_indices)):
        ind = parent_indices[i]
        vals = heap[ind]
        heap[ind] = (vals[0]+update_values[i], vals[1], vals[2])
    hq.heapify(heap)
    return heap

def replace_leaves_with_children(solutions: np.array, heap: list, children: np.array, children_fitness: np.array) -> tuple:
    """Replaces leaf genomes with children (pre-selected by fitness)

    Args:
        solutions (np.array): matrix of genomes
        heap (list): heap in the format of list
        children (np.array): numpy array of child genomes
        children_fitness (np.array): numpy array of child fitness values

    Returns:
        tuple: numpy array of genomes, heap
    """
    n_children: int =  children_fitness.shape[0]
    k_worst_solutions: list = list(map(lambda x: x[1], heap[-n_children:]))

    solutions[k_worst_solutions, :] =  children
    for i in range(n_children):
        heap[-i] = (children_fitness[i], k_worst_solutions[i], children_fitness[i])
    return solutions, heap