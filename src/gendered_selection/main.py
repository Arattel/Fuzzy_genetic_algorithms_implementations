import numpy as np
import pandas as pd
import simpful as sf
from scipy.spatial import distance_matrix
from tqdm import tqdm

from .helpers import (calculate_lifetime)
from ..common.utils import (quadratic_fitness, mutate_single_point, crossover, minus_sign)
from .conf.gendered_selection_config import Config
from .faster_fuzzy_logic.infer_partner_age import Inferrer
import line_profiler

profiler = line_profiler.LineProfiler()


def schwefel(genome):
    return -np.abs(418.9829 * genome.shape[0] - np.sum(genome * np.sin(np.sqrt(np.abs(genome)))))


class Simulation:
    def __init__(self, conf: Config, fitness_fn=None, mutation=None, crossover=None) -> None:
        self.cfg = conf

        # Given that our simulation is geared towards maximization task, we transform a non-negative function to be 
        # minimized to a non-positive one to be maximized 
        self.fitness_fn = minus_sign(fitness_fn)
        self.mutation = mutation
        self.crossover = crossover
    
    @profiler
    def run(self, n_epochs: int =  20, seed: int = 42, percent_males_reproducing=None, population_scale=1,  mutation_scale = .2,) -> None:
        if percent_males_reproducing is not None:
            self.cfg.PERCENT_MALES_REPRODUCING = percent_males_reproducing

        np.random.seed(seed)
        N_FITNESS_FN_CALLS: int = 0
        FS =  Inferrer(rule_path=self.cfg.RULES_FILE)
        
        # Generate initial population
        genomes = np.random.uniform(-population_scale, population_scale, size=(self.cfg.N, 5))
        fitness = np.apply_along_axis(self.fitness_fn, 1, genomes)
        N_FITNESS_FN_CALLS += genomes.shape[0]
        gender = (np.random.rand(self.cfg.N) >= .5).astype(int)
        age = np.zeros(self.cfg.N)
        
        history = []
        
        for epoch in tqdm(range(n_epochs)):
            age += 1
            male_indices = np.argwhere(gender == 1).flatten()
            female_indices = np.argwhere(gender == 0).flatten()
            
            if not (male_indices.shape[0] and female_indices.shape[0]):
                print(f'Early stopping at epoch: {epoch}, population died')
                break
                        
                        
            # If we have a small population, it still reproduces
            to_select = np.ceil(male_indices.shape[0] * self.cfg.PERCENT_MALES_REPRODUCING).astype(int)


            random_males = np.random.choice(male_indices, size=to_select)
            
            lifetime = calculate_lifetime(L=self.cfg.L, U = self.cfg.U, fitness=fitness, age=age)
            diversity = age[male_indices] / lifetime[male_indices]
            population_diversity = diversity.mean()
            
            female_preferred_age = np.array([FS.infer_partner_age(age=lifetime[i], diversity=population_diversity) for i in random_males])
            mate_selection = np.argmin(distance_matrix(female_preferred_age.reshape(-1, 1), lifetime[female_indices].reshape(-1, 1)), axis=1)
            
            
            # Generating children using crossover
            children = np.array([crossover(genomes[random_males[i]], genomes[mate_selection[i]]) for i in range(len(random_males))])
            
            
            # Adding children to the general population
            # 
            
            genomes = np.vstack([genomes, children])
           
            
            age = np.concatenate([age, np.zeros(to_select)])
            gender = np.concatenate([gender, (np.random.rand(children.shape[0]) >= .5).astype(int)])
            
            # Calculating fitness again
            
            genomes = np.apply_along_axis(self.mutation(mutation_scale), 1, genomes)
            fitness = np.apply_along_axis(self.fitness_fn, 1, genomes)
            N_FITNESS_FN_CALLS += genomes.shape[0]
            
            # Removing "old" genomes 
            lifetime = calculate_lifetime(L=Config.L, U = Config.U, fitness=fitness, age=age)
            to_remove = lifetime >= 1.0
            genomes = genomes[~to_remove, :]
            gender =  gender[~to_remove]
            age =  age[~to_remove]
            fitness = fitness[~to_remove]

            history.append({
                'avg_fitness': np.mean(fitness), 
                'max_fitness': np.max(fitness), 
                'population_size': genomes.shape[0], 
                'epoch': epoch
            })
        history =  pd.DataFrame.from_records(history)
        history =  history.rename({'max_fitness': 'best_fitness'}, axis=1)

        # Given that our fitness is non-positive, we transform it into non-negative
        history[['best_fitness', 'avg_fitness']] = -history[['best_fitness', 'avg_fitness']]

        return history, genomes[np.argmax(fitness)], N_FITNESS_FN_CALLS
        

if __name__ == '__main__':
    s =  Simulation(conf=Config(), fitness_fn=schwefel, mutation=mutate_single_point, crossover=crossover)
    s.run(n_epochs=120)
    profiler.print_stats()
