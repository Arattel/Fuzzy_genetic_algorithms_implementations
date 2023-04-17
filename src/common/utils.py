import numpy as np


def quadratic_fitness(vec):
    return np.sum(np.square(vec))


def crossover(male_genome, female_genome):
    i = np.random.randint(low=1,high=male_genome.shape[0]-1)
    return np.concatenate([male_genome[:i], female_genome[i:]])

def crossover_2(p1, p2):
    i = np.random.randint(low=1,high=p1.shape[0]-1)
    return np.concatenate([p1[:i], p2[i:]]),  np.concatenate([p2[:i], p1[i:]])

def mutate_single_point(genome, p_mutation: float=.2):
    if np.random.rand() < p_mutation:
        print(genome)
        k = np.random.randint(low=0, high=genome.shape[0] - 1)
        genome[k] += np.random.normal(loc=0.0, scale=.2)
    return genome


def _mutate_single_point(genome):
    k = np.random.randint(low=0, high=genome.shape[0] - 1)
    genome[k] += np.random.normal(loc=0.0, scale=.2)
    return genome

def mutation(loc,):
    def _mutation(genome):
        return genome + np.random.normal(loc, scale=3)
    return _mutation

def minus_sign(fn):
    def minus_fn(x):
        return -fn(x)
    return minus_fn

def generate_population(seed: int, lower: float, higher: float, N_individuals: int, N_dimensions: int = 5) -> np.array:
    """A unified method of genome generation for all algorithms in order to have the same initial population

    Args:
        seed (int): random seed
        lower (float): lower bound
        higher (float): upper bound
        N_individuals (int): number of individuals
        N_dimensions (int, optional): number of dimensiton. Defaults to 5.

    Returns:
        np.array: population
    """
    np.random.seed(seed)
    return np.random.uniform(lower, higher, size=(N_individuals, N_dimensions))

if __name__ == '__main__':
    print('Hellow')