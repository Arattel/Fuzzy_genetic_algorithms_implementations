import fuzzylite as fl

from .variables import (age, partner_age, diversity)


class Inferrer:
    def __init__(self, rule_path: str) -> None:
        self.engine = fl.Engine(name="approximation", description="")
        self.engine.input_variables = [age, diversity]
        self.engine.output_variables = [partner_age]

        self.age = age
        self.divesity = diversity
        self.partner_age = partner_age

        # Reading rules
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
        ],)]


    def infer_partner_age(self, age, diversity) -> float:
        self.age.value = age
        self.divesity.value = diversity

        self.engine.process()


        return self.partner_age.value



if __name__ == '__main__':
    inf = Inferrer('../conf/rules_ref.txt')
    print(inf.infer_partner_age(.35, 5.9))