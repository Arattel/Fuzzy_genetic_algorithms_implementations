from typing import Dict
from .param_inference import ParamInference

class AllParamInference:
    def __init__(self, param_config: Dict) -> None:
        self._inferrers = {}
        for parameter_name in param_config:
            self._inferrers[parameter_name] = ParamInference(varname=parameter_name, init_fn=param_config[parameter_name]['fn'], 
                                                             rule_path=param_config[parameter_name]['path'])
    
    def infer(self,  bestFitness: float, avgFitness: float, avgFitChange: float) -> Dict[str, float]:
        result = {}

        for parameter_name in self._inferrers:
            result[parameter_name] = self._inferrers[parameter_name].infer(bestFitness, avgFitness, avgFitChange)
        
        return result
