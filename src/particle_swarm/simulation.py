import numpy as np
import pandas as pd

from ..common.utils import generate_population

def simulation(seed: int, N_genomes: int, N_dimensions: int, N_epochs:int, fitness_fn, low: float, high: float,phi_p = .5, phi_g = .3, w = .2, verbose: bool = False):
    np.random.seed(seed)
    N_CALLS: int = 0
    population = generate_population(seed, low, high, N_genomes, N_dimensions)
    best_known = population.copy()
    best_known_fitness = np.apply_along_axis(fitness_fn, 1, population)
    N_CALLS += N_genomes
    swarm_best_known = best_known[np.argmin(best_known_fitness)]
    swarm_best_fitness = best_known_fitness.min()
    velocities = generate_population(seed, -np.abs(high - low), np.abs(high - low), N_genomes, N_dimensions)
    history  = []
    for i in range(N_epochs):
        for particle_index in range(N_genomes):
            r_p, r_g = np.random.rand(N_dimensions), np.random.rand(N_dimensions)
            velocities[particle_index] = w * velocities[particle_index] \
            + phi_p * r_p * (best_known[particle_index] - population[particle_index]) \
            + phi_g * r_g * (swarm_best_known - population[particle_index])
            population[particle_index] = np.minimum(np.maximum(population[particle_index] + velocities[particle_index], low), high)
            # population[particle_index][population[particle_index] < low] = low
            # population[particle_index][population[particle_index] > high] = high


            N_CALLS += 1
            cur_fitness = fitness_fn(population[particle_index])
            if cur_fitness < best_known_fitness[particle_index]:
                best_known[particle_index] = population[particle_index]
                best_known_fitness[particle_index] = cur_fitness
                
                if cur_fitness < swarm_best_fitness:
                    swarm_best_known = best_known[particle_index]
                    swarm_best_fitness = cur_fitness
        N_CALLS += 1
        history.append({'epoch': i, 'best_fitness': swarm_best_fitness})
        if i % 20 and verbose:
            print(f'Best fitness: {swarm_best_fitness}')
    df = pd.DataFrame.from_records(history)
    return df, swarm_best_known, N_CALLS