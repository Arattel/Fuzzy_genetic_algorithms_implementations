import fuzzylite as fl
from .linguistic_variables import *

class ParamInference:
    def __init__(self, rule_path_param: str, rule_path_priority: str) -> None:
        self.engine_params = fl.Engine(name="approximation", description="")
        self.engine_params.input_variables = [average_fitness, best_fitness, average_fitness_change]
        self.engine_params.output_variables = [x_rate, m_rate, subpop_size]

        self.engine_priority = fl.Engine(name="approximation", description="")
        self.engine_priority.input_variables = [first_cl, second_cl]
        self.engine_priority.output_variables = [priority]

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

        rules_params = []
        with open(rule_path_param, 'r') as f:
            rules_params = f.readlines()

        rules_priority = []
        with open(rule_path_priority, 'r') as f:
            rules_priority = f.readlines()

        self.engine_params.rule_blocks = [
            fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.Minimum(),
        disjunction=None,
        implication=fl.Minimum(),
        activation=fl.General(),
        rules=[
    fl.Rule.create(rule.strip(), self.engine_params) for rule in rules_params
        ],
    )]
        self.engine_priority.rule_blocks = [
            fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=fl.Minimum(),
        disjunction=None,
        implication=fl.Minimum(),
        activation=fl.General(),
        rules=[
    fl.Rule.create(rule.strip(), self.engine_priority) for rule in rules_priority
        ],
    )
]

 


    def infer(self, bestFitness: float, avgFitness: float, avgFitChange: float) -> float:
        self.best_fitness.value = bestFitness
        self.average_fitness.value = avgFitness
        self.average_fitness_change.value = avgFitChange

        self.engine_params.process()

        return {'xRate':self.x_rate.value, 'mRate':self.m_rate.value, 'subPopSize': self.subpop_size.value}
       
    def infer_priority(self, first, second):

        if first > second:
            first, second = second, first
        

        self.first_cl.value = first
        self.second_cl.value = second

        self.engine_priority.process()

        return self.priority.value


