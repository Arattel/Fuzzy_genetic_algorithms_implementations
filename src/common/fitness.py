import numpy as np

def schwefel(genome):
    return np.abs(418.9829 * genome.shape[0] - np.sum(genome * np.sin(np.sqrt(np.abs(genome)))))