import numpy as np

class BernoulliBandit:
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx] 
        self.K = K
    def step(self, K):
        return np.random.binomial(1, self.probs[K])
    
class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = []
        self.regrets = []
        
    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
        
    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k =  self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)
            
if __name__ == "__main__":
    K = 10
    bandit = BernoulliBandit(K)
    print("Probabilities of each arm:", bandit.probs)
    print("Best arm index:", bandit.best_idx)
    print("Best arm probability:", bandit.best_prob)
    
    # Test the step function
    for i in range(K):
        reward = bandit.step(i)
        print(f"Reward from arm {i}: {reward}")