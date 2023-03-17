import numpy as np

def schwefel(genome):
    """
    The original Schwefel function has a minimum value of 2094.91. 2094.91 was added to it to make it non-negative. 
    """
    return  2094.91 -np.sum(genome * np.sin(np.sqrt(np.abs(genome))))


def griewank(genome):
    return 1 + (np.sum(np.square(genome)) / 400) - np.product(np.cos(
        genome / (np.arange(genome.shape[0]) + 1)))


def rastrigin(genome):
    return np.sum(
        np.square(genome) - 10 * np.cos(2*np.pi*genome) + 10
    )

def ackley(genome):
    return -20 * np.exp(-0.2 * 
                       np.sqrt(1 /  genome.shape[0]) * 
                       np.sum(np.square(genome))) \
                       - np.exp((1 / genome.shape[0]) * 
                       np.sum(np.cos(2*np.pi * genome))) \
                       + 20 + np.e