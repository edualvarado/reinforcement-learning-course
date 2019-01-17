import numpy as np
import matplotlib.pyplot as plt
import sys 

"""
Implement the different exploration strategies (do not forget the schedule in
line 88).

mab is a MAB (MultiArmedBandit) object as defined below
epsilon is a scalar, which influences the amount of random actions
schedule is a callable decaying epsilon

You can get the approximated Q-values via mab.bandit_q_values and the different
counters for the bandits via mab.bandit_counters.
"""
def epsilon_greedy(mab, epsilon):
  raise NotImplementedError()

def decaying_epsilon_greedy(mab, epsilon, schedule):
  raise NotImplementedError()

def random(mab):
  raise NotImplementedError()

def ucb1(mab):
  raise NotImplementedError()

class Bandit:
  def __init__(self, bias, q_value=0, counter=0):
    self.bias = bias
    self.q_value = q_value
    self.counter = counter

  def pull(self):
    self.counter += 1
    reward = np.clip(self.bias + np.random.uniform(), 0, 1)
    self.q_value = self.q_value + 1/self.counter * (reward - self.q_value)
    return reward

class MAB:
  def __init__(self, best_action, *bandits):
    self.bandits = bandits
    self._no_actions = len(bandits)
    self.step_counter = 0
    self.best_action = best_action

  def pull(self, action):
    self.step_counter += 1
    return self.bandits[action].pull(), self.bandits[action].q_value

  def run(self, no_rounds, exploration_strategy, **strategy_parameters):
    regrets = []
    rewards = []
    for i in range(no_rounds):
      if (i + 1) % 100 == 0:
        print("\rEpisode {}/{}".format(i + 1, no_rounds), end="")
        sys.stdout.flush()
      action = exploration_strategy(self, **strategy_parameters)
      reward, q = self.pull(action)
      best_action_value = self.best_action(self)[1]
      regret = best_action_value - q
      regrets.append(regret)
      rewards.append(reward)
    return regrets, rewards

  @property
  def bandit_counters(self):
    return np.array([bandit.counter for bandit in self.bandits])

  @property
  def bandit_q_values(self):
    return np.array([bandit.q_value for bandit in self.bandits])

  @property
  def no_actions(self):
    return self._no_actions

def plot(regrets):
  for strategy, regret in regrets.items():
    total_regret = np.cumsum(regret)
    plt.ylabel('Total Regret')
    plt.plot(np.arange(len(total_regret)), total_regret, label=strategy)
  plt.legend()
  plt.savefig('regret.png')

if __name__ == '__main__':
  def schedule(mab, epsilon):
    raise NotImplementedError()

  epsilon = 0.5

  strategies = {
    epsilon_greedy: {'epsilon': epsilon},
    decaying_epsilon_greedy: {'epsilon': epsilon, 'schedule': schedule},
    random: {},
    ucb1: {}
  }

  average_total_returns = {}
  total_regrets = {}
  num_actions = 10
  biases = [1.0 / k for k in range(5, 5+num_actions)]
  best_action_index = 0
  best_action_value = 0.7
  def best_action(mab):
    return best_action_index, best_action_value
  for strategy, parameters in strategies.items():
    print(strategy.__name__)
    bandits = [Bandit(bias, 1-bias) for bias in biases]
    mab = MAB(best_action, *bandits)
    total_regret, average_total_return = mab.run(1000000, strategy, **parameters)
    print("\n")
    average_total_returns[strategy.__name__] = average_total_return
    total_regrets[strategy.__name__] = total_regret
  plot(total_regrets)