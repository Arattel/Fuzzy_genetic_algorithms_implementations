import fuzzylite as fl

from ...common.generalized_helpers import generate_var_terms, _generate_bin_name

def calculate_n_priority_bins(n_terms_fitness: int) -> int:
    return 2 * n_terms_fitness - 2 + 1

def generate_rules(n_terms_fitness: int) -> list[str]:
    """Generates rules for priority inference

    Args:
        n_terms_fitness (int): number of terms of fitness

    Returns:
        list[str]: list of rules for priority inference
    """
    n_bins_priority = calculate_n_priority_bins(n_terms_fitness=n_terms_fitness)
    rules = []
    rules_code = {}
    for first_child_fitness in range(n_terms_fitness -1, -1, -1):
        for second_child_fitness in range(n_terms_fitness -1, -1, -1):
            i, j =  first_child_fitness, second_child_fitness
            if first_child_fitness > second_child_fitness:
                i, j = second_child_fitness, first_child_fitness
            if (i, j) not in rules_code:
                rules_code[(i, j)] = n_bins_priority - (first_child_fitness + second_child_fitness) - 1 
    for i in rules_code:
        rule = f'if first is {_generate_bin_name(i[0], n_terms_fitness)} and second is {_generate_bin_name(i[1], n_terms_fitness)}  then priority is {_generate_bin_name(rules_code[i], n_bins_priority)}'
        rules.append(rule)
    
    return rules[::-1]

class GeneralizedPriorityInferencer:
    def __init__(self, n_terms_fitness=3, membership_function='trapezoid', t_conorm=None, t_norm=None) -> None:
        self.n_priority_bins = calculate_n_priority_bins(n_terms_fitness)
        self.membership_function = membership_function
        self.first_cl = fl.InputVariable(
            name="first",
            enabled=True,
            minimum=0.0,
            maximum=1.0,
            lock_range=False,
            terms=generate_var_terms(universe=(0, 1), trapezoid_points=(0, 1), n_terms=n_terms_fitness, 
                                     type=self.membership_function))
        
        self.second_cl = fl.InputVariable(
            name="second",
            enabled=True,
            minimum=0.0,
            maximum=1.0,
            lock_range=False,
            terms=generate_var_terms(universe=(0, 1), trapezoid_points=(0, 1), n_terms=n_terms_fitness, 
                                     type=self.membership_function))
        
        self.priority = fl.OutputVariable(
            name="priority",
            enabled=True,
            minimum=-1,
            maximum=1,
            lock_range=False,
            defuzzifier=fl.Centroid(),
            aggregation=fl.Maximum(),
            terms=generate_var_terms(universe=(-1, 1), trapezoid_points=(-1, 1), n_terms=self.n_priority_bins, 
                                     type=self.membership_function))
        
        self.engine = fl.Engine(name="approximation", description="")
        self.engine.input_variables = [self.first_cl, self.second_cl]
        self.engine.output_variables = [self.priority]


        if t_norm == 'min':
            self.t_norm = fl.Minimum()
        elif t_norm == 'product': 
            self.t_norm = fl.AlgebraicProduct()

        if t_conorm == 'max': 
            self.t_conorm = fl.Maximum()
        elif t_conorm == 'sum':
            self.t_conorm = fl.AlgebraicSum()


        self.engine.rule_blocks = [
            fl.RuleBlock(
            name="",
            description="",
            enabled=True,
            conjunction=self.t_norm,
            disjunction=self.t_conorm, 
            implication=fl.Minimum(),
            activation=fl.General(),
            rules= [fl.Rule.create(rule.strip(), self.engine) for rule in generate_rules(n_terms_fitness=n_terms_fitness)])]
    

    def infer_priority(self, first, second):

        if first > second:
            first, second = second, first
        

        self.first_cl.value = first
        self.second_cl.value = second

        self.engine.process()

        return self.priority.value

        