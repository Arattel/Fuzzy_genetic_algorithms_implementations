import numpy as np
import fuzzylite as fl
from matplotlib import pyplot as plt


def plot_variable(var, universe=None):
    if universe is None:
        universe = (var.minimum, var.maximum)
    plt.title(var.name)
    linguistic_terms_names = []
    vals = np.linspace(*universe, 100)
    
        
    for term in var.terms:
        membership_fn = [term.membership(e) for e in vals]
        plt.plot(vals, membership_fn)
        linguistic_terms_names.append(term.name)
    plt.legend(linguistic_terms_names)
    plt.xlabel(var.name)
    plt.ylabel('membership function')
    plt.show()

def _generate_bin_name(index: int, n_bins: int) -> str:
    cat_name = None
    if index == 0:
        cat_name = 'first_bin'
    elif index < n_bins - 1:
        cat_name = f'bin_{index+1}'
    elif index == n_bins - 1:
        cat_name = f'last_bin'
    return cat_name



def generate_var_terms(universe: tuple[float], trapezoid_points: tuple[float], n_terms: int = 4) -> list[fl.Term]:
    """Generates terms given universe, trapezoid points and number of term

    Args:
        universe (tuple[float]): general universe of the variable
        trapezoid_points (tuple[float]): left/right 
        n_terms (int, optional): Number of terms. Defaults to 4.

    Returns:
        list[fl.Term]: list of terms
    """
    points = np.linspace(trapezoid_points[0], trapezoid_points[1], n_terms)
    terms = [fl.Trapezoid("first_bin", universe[0] - 1, universe[0], points[0], points[1]),
             fl.Trapezoid("last_bin", points[-2], points[-1], universe[1], universe[1] + 1),]
    
    if len(points) > 2: 
        for i in range(n_terms - 2):
            index = i + 2
            term = fl.Triangle(f'bin_{index}', points[i], points[i + 1], points[i + 2])
            terms.append(term)
    return terms