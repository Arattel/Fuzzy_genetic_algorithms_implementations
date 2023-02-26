import fuzzylite as fl
from .linguistic_variables import *

class ParamInference:
    def __init__(self, rule_path: str) -> None:
        self.engine = fl.Engine(name="approximation", description="")
        self.engine.input_variables = [average_fitness, best_fitness, average_fitness_change, first_cl, second_cl]
        self.engine.output_variables = [x_rate, m_rate, subpop_size, priority]

        # Input variables
        self.average_fitness = average_fitness
        self.best_fitness =  best_fitness
        self.average_fitness_change = average_fitness_change
        self.first_cl = first_cl
        self.second_cl = second_cl

        # Output
        self.x_rate = x_rate
        self.m_rate = m_rate
        self.subpop_size = subpop_size
        self.priority =  priority

        rules = []
        with open(rule_path, 'r') as f:
            rules = f.readlines()

        self.engine.rule_blocks = [
            fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.Minimum(),
        disjunction=None,
        implication=fl.Minimum(),
        activation=fl.General(),
        rules=[
    fl.Rule.create(rule.strip(), self.engine) for rule in rules
        ],
    )
]

 


    def infer(self, bestFitness: float, avgFitness: float, avgFitChange: float) -> float:
        self.best_fitness.value = bestFitness
        self.average_fitness.value = avgFitness
        self.average_fitness_change.value = avgFitChange

        self.engine.process()

        return {'xRate':self.x_rate.value, 'mRate':self.m_rate.value, 'subPopSize': self.subpop_size.value}
       
    def infer_priority(self, first, second):

        if first > second:
            first, second = second, first
        

        self.first_cl.value = first
        self.second_cl.value = second

        self.engine.process()

        return self.priority.value


