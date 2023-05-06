import heapq as hq
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

from src.elegant_fuzzy_genetic_algorithms.helpers.heap_helpers import (create_heap, replace_leaves_with_children, 
                                                                       approximate_topk_indices, update_parents_with_indices)
from src.elegant_fuzzy_genetic_algorithms.helpers.all_params_wrapper import AllEFGAParamsParallelWrapper
from src.elegant_fuzzy_genetic_algorithms.conf.param_inference_config import Conf
from src.common.utils import (quadratic_fitness, _mutate_single_point, crossover_2, mutate_single_point, crossover, mutation, 
                              generate_population)
from src.gendered_selection.main import Simulation
from src.gendered_selection.conf.gendered_selection_config import Config as GenderedSelectionCOnfig

class SimulationConfig: 
    mutation_scale: float = 2
    x_range: float = 2000

def simulation(  N = 50, epochs: int =  100, verbose = False, default_params = Conf.default_params, conf: SimulationConfig = SimulationConfig(), fitness_fn = None, 
               mutation_scale = None, population_scale=None, seed=42,  n_terms_params: int = 3, n_terms_priority: int = 3, ndim: int = 5, membership_function = 'trapezoid', 
               use_approx = True):
    np.random.seed(seed)
    if mutation_scale is not None:
        conf.mutation_scale = mutation_scale

    if population_scale is not None:
        conf.x_range = population_scale
    
    priority_inferencer =  AllEFGAParamsParallelWrapper(n_terms_params=n_terms_params, n_terms_priority=n_terms_priority, membership_function=membership_function, 
                                                         use_approx=use_approx)

    N_FITNESS_FN_CALLS: int = 0
    params = default_params
    n_subpop_individuals = np.round(N * params['subPopSize']).astype(int)
    
    
    genomes =  generate_population(lower=-conf.x_range, higher=conf.x_range, 
                                   N_individuals=N, 
                                   N_dimensions=ndim, 
                                   seed=seed)
    
    fitness = np.apply_along_axis(fitness_fn, 1, genomes)
    N_FITNESS_FN_CALLS += genomes.shape[0]
    worst_initial_fitness = np.max(fitness)
    fitness = fitness / worst_initial_fitness
    heap = create_heap(fitness)
    best_fitness = heap[0][2]
    history = []
    
    for i in tqdm(range(epochs)):
        print(i)
        # Indices of parents in heap & fitness
        topk = np.arange(n_subpop_individuals)
        subpop_fitness = np.array([heap[topk[x]][2] for x in range(n_subpop_individuals)])
        # Indices of parents in genome array    
        topk_indices = [heap[topk[x]][1] for x in range(n_subpop_individuals)]
        
        # Calculate best fitness
        avg_fitness = np.mean(subpop_fitness)
        
        # In case topk is uneven we take the even subpop
        # 
        crossover_indices = topk[:(n_subpop_individuals // 2) * 2]
        
        # Shuffle & select pairs for crossover
        np.random.shuffle(crossover_indices)
        parent_1, parent_2 = crossover_indices[::2], crossover_indices[1::2]
        do_crossover = np.random.rand(parent_1.shape[0]) < params['xRate']

        while not do_crossover.any():
            do_crossover = np.random.rand(parent_1.shape[0]) < params['xRate']

        
        # Parent 1 and parent 2 to do crossover
        # 
        parent_1 = parent_1[do_crossover]
        parent_2 = parent_2[do_crossover]
        
        child_genomes = []
        for j in range(parent_1.shape[0]):
            # Getting genomes of the parent pair
            p1 = genomes[heap[parent_1[j]][1]]
            p2 = genomes[heap[parent_2[j]][1]]
            
            c1, c2 = crossover_2(p1, p2)
            child_genomes += [c1, c2]
        child_genomes = np.array(child_genomes)
        
        # Mutation
        mutate = np.random.rand(child_genomes.shape[0]) <= params['mRate']


        if mutate.sum():
            child_genomes[mutate] =  np.clip(np.apply_along_axis(mutation(conf.mutation_scale), 1, child_genomes[mutate]), 
                                             -population_scale, population_scale)
            
            
        # Calculating children fitness, replacing worst solutions with good childre
        prev_avg_fitness = np.mean([x[2] for x  in heap])
        child_fitness = np.apply_along_axis(fitness_fn, 1, child_genomes) / worst_initial_fitness
        N_FITNESS_FN_CALLS += child_genomes.shape[0]
        use_to_replace = child_fitness < avg_fitness
        
        if use_to_replace.sum():
            genomes, heap = replace_leaves_with_children(solutions=genomes, heap=heap, children=child_genomes[use_to_replace], children_fitness=child_fitness[use_to_replace])
    
        c1_fitness, c2_fitness = child_fitness[::2], child_fitness[1::2]
        priority_updates = np.repeat(-priority_inferencer.infer_priority(c1_fitness, c2_fitness), 2)
        parent_indices = np.zeros(parent_1.shape[0] * 2, dtype=int)
        parent_indices[::2] =  parent_1
        parent_indices[1::2] = parent_2

        mutate = np.random.rand(parent_indices.shape[0]) < params['mRate']

    
        for j in range(len(parent_indices)):
            if mutate[j]:
                index_to_mutate = heap[parent_indices[j]][1]
                genomes[index_to_mutate] = np.clip(mutation(conf.mutation_scale)(genomes[index_to_mutate]), -population_scale, population_scale)
        
        
        # Parent priorities updated, heap heapified
        heap = update_parents_with_indices(heap=heap, parent_indices=parent_indices, update_values=priority_updates)
        hq.heapify(heap)
    
        cur_avg_fitness = np.mean([x[2] for x in heap])
        avg_fit_change = np.abs(prev_avg_fitness - cur_avg_fitness)
        
        params = priority_inferencer.infer(bestFitness=best_fitness, avgFitness=avg_fitness, avgFitChange=avg_fit_change)

        if verbose:
            print(f'Best solution: {genomes[heap[0][1]]}, fitness: {heap[0][2]}, avg fitness: {cur_avg_fitness}')
        
        history.append({'best_fitness': fitness_fn(genomes[heap[0][1]])/worst_initial_fitness, 'avg_fitness': cur_avg_fitness, 
                        'epoch': i})
    
    df = pd.DataFrame.from_records(history)

    # Transform relative fitness into absolute fitness
    df[['best_fitness', 'avg_fitness']] = df[['best_fitness', 'avg_fitness']] * worst_initial_fitness

    return df, genomes[heap[0][1]], N_FITNESS_FN_CALLS