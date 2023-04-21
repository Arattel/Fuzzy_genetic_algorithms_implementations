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


# Source: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays/1235363#1235363
def cartesian(arrays, out=None):
    """
    Generate a Cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the Cartesian product of.
    out : ndarray
        Array to place the Cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing Cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out