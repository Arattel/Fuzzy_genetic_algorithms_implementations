import numpy as np
import pandas as pd
import simpful as sf
from scipy.spatial import distance_matrix
from tqdm import tqdm

from .helpers import (calculate_lifetime, init_age, init_diversity, calculate_partner_age, init_fuzzy_system)
from ..common.utils import (quadratic_fitness, mutate_single_point, crossover)
from .conf.gendered_selection_config import Config


class Simulation:
    def __init__(self, conf: Config, fitness_fn=None, mutation=None, crossover=None) -> None:
        self.cfg = conf
        self.fitness_fn = fitness_fn
        self.mutation = mutation
        self.crossover = crossover
    
    def run(self, n_epochs: int =  20) -> None:
        FS =  init_fuzzy_system(rules_file=self.cfg.RULES_FILE)
        
        # Generate initial population
        genomes = np.random.normal(loc=2, scale=3, size=(self.cfg.N, 5))
        fitness = np.apply_along_axis(self.fitness_fn, 1, genomes)
        gender = (np.random.rand(self.cfg.N) >= .5).astype(int)
        age = np.zeros(self.cfg.N)
        
        history = []
        
        for epoch in tqdm(range(n_epochs)):
            age += 1
            male_indices = np.argwhere(gender == 1).flatten()
            female_indices = np.argwhere(gender == 0).flatten()
            
            to_select = np.round(male_indices.shape[0] * self.cfg.PERCENT_MALES_REPRODUCING).astype(int)
            random_males = np.random.choice(male_indices, size=to_select)
            
            lifetime = calculate_lifetime(L=self.cfg.L, U = self.cfg.U, fitness=fitness, age=age)
            diversity = age[male_indices] / lifetime[male_indices]
            population_diversity = diversity.mean()
            
            female_preferred_age = np.array([calculate_partner_age(FS, age=lifetime[i], diversity=population_diversity) for i in random_males])
            mate_selection = np.argmin(distance_matrix(female_preferred_age.reshape(-1, 1), lifetime[female_indices].reshape(-1, 1)), axis=1)
            
            
            # Generating children using crossover
            children = np.array([crossover(genomes[random_males[i]], genomes[mate_selection[i]]) for i in range(len(random_males))])
            
            
            # Adding children to the general population
            # 
            genomes = np.vstack([genomes, children])
            age = np.concatenate([age, np.zeros(to_select)])
            gender = np.concatenate([gender, (np.random.rand(children.shape[0]) >= .5).astype(int)])
            
            # Calculating fitness again
            
            genomes = np.apply_along_axis(self.mutation, 1, genomes)
            fitness = np.apply_along_axis(quadratic_fitness, 1, genomes)
            
            # Removing "old" genomes 
            lifetime = calculate_lifetime(L=Config.L, U = Config.U, fitness=fitness, age=age)
            to_remove = lifetime >= 1.0
            genomes = genomes[~to_remove, :]
            gender =  gender[~to_remove]
            age =  age[~to_remove]
            fitness = fitness[~to_remove]

            history.append({
                'median_fitness': np.median(fitness), 
                'max_fitness': np.max(fitness), 
                'population_size': genomes.shape[0], 
                'epoch': epoch
            })
        history =  pd.DataFrame.from_records(history)
        history.to_csv('./log/history.csv')
        

if __name__ == '__main__':
    s =  Simulation(conf=Config(), fitness_fn=quadratic_fitness, mutation=mutate_single_point, crossover=crossover)
    s.run()
