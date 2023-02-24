from typing import Dict, Any
from src.elegant_fuzzy_genetic_algorithms.helpers.linguistic_variables import (init_x_rate, init_m_rate, init_subpop_size)

parameters_to_infer: Dict[str, Any] = {
    'xRate': {
    'fn': init_x_rate, 
    'path': 'src/elegant_fuzzy_genetic_algorithms/rulesets/ruleset_Xrate.txt'
    }, 
    'mRate': {
        'fn': init_m_rate, 
        'path': 'src/elegant_fuzzy_genetic_algorithms/rulesets/ruleset_mrate.txt'
    }, 
    'subPopSize': {
        'fn': init_subpop_size, 
        'path': 'src/elegant_fuzzy_genetic_algorithms/rulesets/ruleset_subpopsize.txt'
    }, 
}
