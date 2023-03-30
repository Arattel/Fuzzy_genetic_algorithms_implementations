import fuzzylite as fl
import numpy as np

from ...common.generalized_helpers import (generate_var_terms, _generate_bin_name)

def generate_rules_parameter_inference(n_terms: int, params_to_update: list[str] = ['xRate', 'mRate', 'subPopSize']) -> list[str]:
    rules = []

    for param in params_to_update:
        rule = f'if  bestFitness is last_bin  then  {param} is last_bin'
        rules.append(rule)
        rule = f'if  avgFitness is first_bin  then  {param} is first_bin'
        rules.append(rule)
    
    for avg_fit_change in range(n_terms):
        for avg_fitness in range(n_terms - 1, 0, -1):
            if avg_fit_change < np.ceil(n_terms / 2):
                index = avg_fitness
            else:
                index = avg_fitness - 1
            rule_bins = (_generate_bin_name(avg_fit_change, n_bins=n_terms), 
              _generate_bin_name(avg_fitness, n_bins=n_terms),
              _generate_bin_name(index, n_bins=n_terms) )
            
            for param in params_to_update:
                rule = f'if  avgFitChange is {rule_bins[0]}   and  avgFitness is {rule_bins[1]}  then  {param} is {rule_bins[2]}'
                rules.append(rule)
    return rules



class GeneralizedParamInferencer:
    def __init__(self, n_terms: int = 3) -> None:
        self.n_terms = n_terms

        self.best_fitness = fl.InputVariable(
            name="bestFitness",
            enabled=True,
            minimum=0.0,
            maximum=.5,
            lock_range=False,
            terms=generate_var_terms(universe=(0, .5), trapezoid_points=(0, .5), n_terms=n_terms),)
        
        self.average_fitness = fl.InputVariable(
            name="avgFitness",
            description="",
            enabled=True,
            minimum=0.0,
            maximum=1.0,
            lock_range=False,
            terms=generate_var_terms(universe=(0, 1), trapezoid_points=(0, 1), n_terms=n_terms),)
        
        self.average_fitness_change =  fl.InputVariable(
            name="avgFitChange",
            description="",
            enabled=True,
            minimum=0.0,
            maximum=1.0,
            lock_range=False,
            terms=generate_var_terms(universe=(0, 1), trapezoid_points=(0, 1), n_terms=n_terms),)
        
        self.x_rate = fl.OutputVariable(
            name="xRate",
            enabled=True,
            minimum=0.4,
            maximum=1.0,
            lock_range=False,
            defuzzifier=fl.Centroid(),
            aggregation=fl.Maximum(),
            terms=generate_var_terms(universe=(0.4, 1), trapezoid_points=(0.4, 1), n_terms=n_terms),)
        
        self.m_rate = fl.OutputVariable(
            name="mRate",
            enabled=True,
            minimum=0,
            maximum=.05,
            lock_range=False,
            defuzzifier=fl.Centroid(),
            aggregation=fl.Maximum(),
            terms=generate_var_terms(universe=(0, .05), trapezoid_points=(0, .05), n_terms=n_terms),)
        

        self.subpop_size = fl.OutputVariable(
            name="subPopSize",
            enabled=True,
            minimum=.14,
            maximum=.26,
            lock_range=False,
            defuzzifier=fl.Centroid(),
            aggregation=fl.Maximum(),
            terms=generate_var_terms(universe=(0.14, .26), trapezoid_points=(0, .26), n_terms=n_terms))
        

        self.engine = fl.Engine(name="approximation", description="")
        self.engine.input_variables = [self.average_fitness, self.best_fitness, self.average_fitness_change]
        self.engine.output_variables = [self.x_rate, self.m_rate, self.subpop_size]

        self.engine.rule_blocks = [
            fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.Minimum(),
        disjunction=None,
        implication=fl.Minimum(),
        activation=fl.General(),
        rules=[fl.Rule.create(rule.strip(), self.engine) for rule in generate_rules_parameter_inference(n_terms=n_terms)],
        )]

    def infer(self, bestFitness: float, avgFitness: float, avgFitChange: float) -> float:
        self.best_fitness.value = bestFitness
        self.average_fitness.value = avgFitness
        self.average_fitness_change.value = avgFitChange

        self.engine.process()

        return {'xRate':self.x_rate.value, 'mRate':self.m_rate.value, 'subPopSize': self.subpop_size.value}
        

    

