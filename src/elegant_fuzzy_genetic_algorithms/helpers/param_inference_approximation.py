import pickle as pkl
import numpy as np
import faiss

from ...common.approximation_helpers import estimate_by_index

class ParamInferenceApprox:
    def __init__(self, dir: str = './indices/') -> None:
        self.dir = dir

        self.y = {}

        for param in ['xRate', 'mRate', 'subPopSize']:
            with open(f'./indices/xgb_{param}_y_approx.pkl', 'rb') as f:
                self.y[param] =  np.array(pkl.load(f))

        self.index  =  faiss.read_index('./indices/params_approx.index')
 

    def infer(self, bestFitness: float, avgFitness: float, avgFitChange: float) -> float:
        res = {}
        for param in self.y:
            q = np.array([[bestFitness, avgFitness, avgFitChange]])
            res[param] = estimate_by_index(self.index, self.y[param], q)[0, 0]
        # print(res)
        return res


