
import fuzzylite as fl
import numpy as np

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

def check_bin_sizes_gendered(age_bins: int = 4, diversity_bins: int = 4):
    return age_bins ==  diversity_bins

def _generate_bin_name(index: int, n_bins: int) -> str:
    cat_name = None
    if index == 0:
        cat_name = 'first_bin'
    elif index < n_bins - 1:
        cat_name = f'bin_{index+1}'
    elif index == n_bins - 1:
        cat_name = f'last_bin'
    return cat_name

def _format_rule(cat1:str, cat2:str, cat3:str) -> str:
    return f'if age is {cat1} and diversity is {cat2} then partner_age is {cat3}'


def generate_gendered_rules(n_bins: int =  4) -> list[str]:
    """Generates rules for partner age given number of bins
    Args:
        n_bins (int, optional): _description_. Defaults to 4.

    Returns:
        list[str]: rules
    """
    rules = []
    for i in range(n_bins):
        age_cat_name: str = _generate_bin_name(i, n_bins)
        for j in range(n_bins):
            diversity_cat_name: str = _generate_bin_name(j, n_bins)

            if i != n_bins - 1:
                age_cat_partner_index = n_bins - j - 1

                if j != n_bins - 1:
                    partner_age_names = _generate_bin_name(age_cat_partner_index, n_bins), _generate_bin_name(age_cat_partner_index - 1, n_bins)
                    for p_age_name in partner_age_names:
                        rule = _format_rule(age_cat_name, diversity_cat_name, p_age_name)
                        rules.append(rule)
                else:
                    partner_age_names = _generate_bin_name(age_cat_partner_index, n_bins)
                    rule = _format_rule(age_cat_name, diversity_cat_name, partner_age_names)
                    rules.append(rule)
            else:

                if j < n_bins - 2:
                    age_cat_partner_index = n_bins - j - 2
                    partner_age_names = _generate_bin_name(age_cat_partner_index, n_bins), _generate_bin_name(age_cat_partner_index - 1, n_bins)
                    for p_age_name in partner_age_names:
                        rule = _format_rule(age_cat_name, diversity_cat_name, p_age_name)
                        rules.append(rule)
                else:
                    age_cat_partner_index = 0
                    partner_age_names = _generate_bin_name(age_cat_partner_index, n_bins)
                    rule = _format_rule(age_cat_name, diversity_cat_name, partner_age_names)
                    rules.append(rule)
    return rules


class GeneralizedInferrer(object):
    def __init__(self, n_partitions: int) -> None:
        self.age = fl.InputVariable('age', enabled=True, minimum=0, maximum=1, 
                           terms=generate_var_terms(universe=(0, 1), trapezoid_points=(.25, .85), n_terms=n_partitions))
        
        self.diversity =  fl.InputVariable('diversity', enabled=True, minimum=0, maximum=10, 
                           terms=generate_var_terms(universe=(0, 10), trapezoid_points=(2.5, 8.5), n_terms=n_partitions))
        self.partner_age = fl.OutputVariable(
            'partner_age',
            enabled=True,
            minimum=0.0,
            maximum=1.0, 
            terms=generate_var_terms(universe=(0, 1), trapezoid_points=(.25, .85), n_terms=n_partitions),
            defuzzifier=fl.Centroid(),
            aggregation=fl.Maximum()
        )
        self.engine = fl.Engine(name="approximation", description="")
        self.engine.input_variables = [self.age, self.diversity]
        self.engine.output_variables = [self.partner_age]

        self.engine.rule_blocks = [
            fl.RuleBlock(
            name="",
            description="",
            enabled=True,
            conjunction=fl.Minimum(),
            disjunction=fl.Maximum(),
            implication=fl.Minimum(),
            activation=fl.General(),
            rules= [fl.Rule.create(rule.strip(), self.engine) for rule in generate_gendered_rules(n_bins=n_partitions)])]
    
    def infer_partner_age(self, age, diversity) -> float:
        self.age.value = age
        self.diversity.value = diversity

        self.engine.process()


        return self.partner_age.value