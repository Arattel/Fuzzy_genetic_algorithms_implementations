import numpy as np


def quadratic_fitness(vec):
    return -np.sum(np.square(vec))


def crossover(male_genome, female_genome):
    i = np.random.randint(low=1,high=male_genome.shape[0]-1)
    return np.concatenate([male_genome[:i], female_genome[i:]])


def mutate_single_point(genome, p_mutation: float=.2):
    if np.random.rand() < p_mutation:
        k = np.random.randint(low=0, high=genome.shape[0] - 1)
        genome[k] += np.random.normal(loc=0.0, scale=.2)
    return genome

if __name__ == '__main__':
    print('Hellow')