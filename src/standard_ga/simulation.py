import random
import numpy as np
import pandas as pd

from ..common.fitness import griewank
from ..common.utils import mutation, crossover_2, generate_population




class GA:
    """
    Genetic algorithm implementation
    """
    
    def __init__(self, fitness_fn, population_scale: float = 600, mutation_scale: float =  20, ndim: int = 5):
        self.fitness = fitness_fn
        self.population_scale = population_scale
        self.mutation_scale = mutation_scale
        self.ndim = ndim
    
    
    def _random_genome(self):
        genome = np.random.uniform(low=-self.population_scale, high=self.population_scale, size=self.ndim)
        return genome
    
    def _mutate(self, genome):
        genome += np.random.normal(size=self.ndim) * self.mutation_scale
        return genome
        
    
    
    def mutate(self, genome, probability=.3):
        if random.random() <= probability:
            return mutation(self.mutation_scale)(genome)
        return genome
    
    def run(self, num_epochs, population_size = 1000, top_percent=.4, verbose=False, seed: int =  42):
        np.random.seed(seed)
        
        population = generate_population(lower=-self.population_scale, higher=self.population_scale, 
                                         N_individuals=population_size, 
                                         N_dimensions=self.ndim, seed=seed)
        history = []
        ncalls = 0
        for epoch in range(num_epochs):
            fitness = -np.apply_along_axis(self.fitness, 1, population)
            ncalls += fitness.shape[0]
            topk = np.argsort(fitness)[::-1][:int(top_percent * population_size)]

            # print(topk, population)
            
            
            reproduction_group = population[topk, :]
            random.shuffle(reproduction_group)
            
            children = []
            for i in range(population_size - len(reproduction_group)):
                p1, p2 = random.randint(0, len(reproduction_group) - 1), random.randint(0, len(reproduction_group) - 1)
                children += crossover_2(reproduction_group[p1], reproduction_group[p2])
            

            children = np.array(children)
            # print(reproduction_group.shape, children.shape)
            population = np.vstack([reproduction_group, children])
            population = np.apply_along_axis(self.mutate, 1, population)

            history.append({'best_fitness': -np.max(fitness), 'avg_fitness': -np.mean(fitness), 
                        'epoch': epoch, 'ncalls': ncalls, 'seed': seed})
            if verbose: 
                print(f'Epoch: {epoch}, Min griewank_function: {-np.max(fitness)}')
                    
                    
                    
        fitness = -np.apply_along_axis(self.fitness, 1, population)
        ncalls += fitness.shape[0]

        df = pd.DataFrame.from_records(history)
        return population[np.argmax(fitness)], df