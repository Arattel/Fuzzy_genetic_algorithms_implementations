import simpful as sf

from src.elegant_fuzzy_genetic_algorithms.helpers.linguistic_variables import init_cf, init_priority

class PriorityInference:
    def __init__(self, rule_path: str) -> None:
        FS = sf.FuzzySystem(show_banner=False)
        FS.add_linguistic_variable("first", init_cf())
        FS.add_linguistic_variable("second", init_cf())
        FS.add_linguistic_variable("priority", init_priority())
        FS.add_rules_from_file(rule_path)
        self._fs = FS
    
    def infer_priority(self, first, second):
        # All rules are written given assumption that fitness of the first is better (which means smaller)
        if first > second:
            first, second =  second, first
        
        self._fs.set_variable('first', first)
        self._fs.set_variable('second', second)
        return self._fs.inference()['priority']
        
