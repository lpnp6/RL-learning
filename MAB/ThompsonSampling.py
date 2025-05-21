from MAB import Solver
import numpy as np


class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self.alpha = np.ones(self.bandit.K)
        self.beta = np.ones(self.bandit.K)

    def run_one_step(self):
        k = np.argmax(np.random.beta(self.alpha, self.beta))
        r = self.bandit.step(k)
        self.alpha[k] += r
        self.beta[k] += 1 - r
        return k
