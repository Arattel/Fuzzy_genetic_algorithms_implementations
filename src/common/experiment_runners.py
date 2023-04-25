import pandas as pd
from tqdm import tqdm


from .utils import (mutate_single_point, crossover, mutation)
from ..gendered_selection.main import Simulation
from ..gendered_selection.conf.gendered_selection_config import Config as GenderedSelectionConfig
from ..elegant_fuzzy_genetic_algorithms.simulation import simulation as efga_simulation
from .fitness import schwefel


def run_experiments_efga(n_experiments: int = 5, fitness_fn=schwefel, population_scale=500, mutation_scale=.5, N=500, epochs=500, n_terms_params: int = 3, n_terms_priority: int = 3, use_approx: bool = True):
    experiment_logs = []
    for experiment in tqdm(range(n_experiments)):
        history, best_solution, ncalls = efga_simulation(epochs=epochs, fitness_fn=fitness_fn, population_scale=population_scale, mutation_scale=mutation_scale, N=N, 
                                                         seed=experiment, n_terms_params=n_terms_params, n_terms_priority=n_terms_priority, use_approx=use_approx)
        history['ncalls'] = ncalls
        history['seed'] = experiment
        experiment_logs.append(history)
    experiment_logs = pd.concat(experiment_logs)
    return experiment_logs, {'n_experiments':  n_experiments, 'fitness_fn': fitness_fn.__name__, 
                             'population_scale': population_scale, 'mutation_scale': mutation_scale, 'N': N, 'epochs': epochs, 
                             'n_terms_params': n_terms_params, 'n_terms_priority': n_terms_priority, 'use_approx': use_approx}


def run_experiments_gendered(n_experiments: int = 5, fitness_fn=schwefel, population_scale=500, mutation_scale=.5, N=500, epochs=500, n_partitions: int = 5):
    experiment_logs = []
    for experiment in tqdm(range(n_experiments)):
        sim =  Simulation(conf=GenderedSelectionConfig(N=N), fitness_fn=fitness_fn, 
                  mutation=mutation, crossover=crossover)
        history, best_solution, ncalls = sim.run(epochs, percent_males_reproducing=.31, population_scale=population_scale, mutation_scale=mutation_scale, seed=experiment, n_partitions=n_partitions)
        history['ncalls'] = ncalls
        history['seed'] = experiment
        experiment_logs.append(history)
    experiment_logs = pd.concat(experiment_logs)
    return experiment_logs, {'n_experiments':  n_experiments, 'fitness_fn': fitness_fn.__name__, 
                             'population_scale': population_scale, 'mutation_scale': mutation_scale, 'N': N, 'epochs': epochs, 'n_partitions': n_partitions}