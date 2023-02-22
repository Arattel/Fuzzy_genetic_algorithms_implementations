import simpful as sf
from .linguistic_variables import (init_best_fitness, init_avg_fitness, init_avg_fit_change)

class ParamInference:
    def __init__(self, varname: str, init_fn, rule_path: str) -> None:
        self.varname = varname
        self.FS = sf.FuzzySystem()
        self.FS.add_linguistic_variable("bestFitness", init_best_fitness())
        self.FS.add_linguistic_variable("avgFitness", init_avg_fitness())
        self.FS.add_linguistic_variable("avgFitChange",init_avg_fit_change())
        self.FS.add_linguistic_variable(varname, init_fn())
        self.FS.add_rules_from_file(rule_path)
    
    def infer(self, bestFitness: float, avgFitness: float, avgFitChange: float) -> float:
        self.FS.set_variable("bestFitness", bestFitness) 
        self.FS.set_variable("avgFitness", avgFitness) 
        self.FS.set_variable("avgFitChange", avgFitChange)
        return self.FS.Mamdani_inference([self.varname], verbose=False)[self.varname]