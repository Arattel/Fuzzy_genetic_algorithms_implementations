from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class Conf: 
    rule_path_param: str = 'src/elegant_fuzzy_genetic_algorithms/rulesets/ruleset_params.txt'
    rule_path_priority: str = 'src/elegant_fuzzy_genetic_algorithms/rulesets/ruleset_priorityt.txt'

    default_params = {'xRate': 0.7,
          'mRate': 0.025,
          'subPopSize': 0.2}
