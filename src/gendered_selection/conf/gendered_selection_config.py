from dataclasses import dataclass

@dataclass
class Config: 
    L:int = 2
    U:int = 10
    N: int = 100
    PERCENT_MALES_REPRODUCING: float = .3
    RULES_FILE: str = 'src/gendered_selection/conf/rules_ref.txt'