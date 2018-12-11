import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
import itertools
import pandas as pd
from PIL import Image

from mountain_car import MountainCarEnv

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
VALID_ACTIONS = [0, 1, 2]

class NeuralNetwork():
  """
  Neural Network class based on TensorFlow.
  """
  def __init__(self):
    self._build_model()

  def _build_model(self):
    """
    Creates a neural network, e.g. with two
    hidden fully connected layers and 20 neurons each). The output layer
    has #A neurons, where #A is the number of actions and has linear activation.
    Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with
    a learning rate of 0.0005). For initialization, you can simply use a uniform
    distribution (-0.5, 0.5), or something different.
    """
    self.states_pl = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    # The TD target value
    self.targets_pl = tf.placeholder(shape=[None], dtype=tf.float32)
    # Integer id of which action was selected
    self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32)

    batch_size = tf.shape(self.states_pl)[0]

    self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, 20, weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.predictions = tf.contrib.layers.fully_connected(self.fc2, len(VALID_ACTIONS), activation_fn=None, weights_initializer=tf.random_uniform_initializer(0, 0.5))

    # Get the predictions for the chosen actions only
    gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
    self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

    # Calcualte the loss
    self.losses = tf.squared_difference(self.targets_pl, self.action_predictions)
    self.loss = tf.reduce_mean(self.losses)

    # Optimizer Parameters from original paper
    #self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
    self.optimizer = tf.train.AdamOptimizer(0.0005)
    self.train_op = self.optimizer.minimize(self.loss)
  
  def predict(self, sess, states):
    """
    Args:
      sess: TensorFlow session
      states: array of states for which we want to predict the actions.
    Returns:
      The prediction of the output tensor.
    """
    prediction = sess.run(self.predictions, { self.states_pl: states })
    return prediction

  def update(self, sess, states, actions, targets):
    """
    Updates the weights of the neural network, based on its targets, its
    predictions, its loss and its optimizer.
    
    Args:
      sess: TensorFlow session.
      states: [current_state] or states of batch
      actions: [current_action] or actions of batch
      targets: [current_target] or targets of batch
    """
    feed_dict = { self.states_pl: states, self.targets_pl: targets,
    self.actions_pl: actions}
    loss = sess.run([self.train_op, self.loss], feed_dict)
    return loss

class TargetNetwork(NeuralNetwork):
  """
  Slowly updated target network. Tau indicates the speed of adjustment. If 1,
  it is always set to the values of its associate.
  """
  def __init__(self, tau=0.001):
    NeuralNetwork.__init__(self)
    self.tau = tau
    self._associate = self._register_associate()

  def _register_associate(self):
    tf_vars = tf.trainable_variables()
    total_vars = len(tf_vars)
    op_holder = []
    for idx,var in enumerate(tf_vars[0:total_vars//2]):
      op_holder.append(tf_vars[idx+total_vars//2].assign(
        (var.value()*self.tau) + 
          ((1-self.tau)*tf_vars[idx+total_vars//2].value())))
    return op_holder
      
  def update(self, sess):
    for op in self._associate:
      sess.run(op)

class ReplayBuffer:
  #Replay buffer for experience replay. Stores transitions.
  def __init__(self):
    self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
    self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])

  def add_transition(self, state, action, next_state, reward, done):
    self._data.states.append(state)
    self._data.actions.append(action)
    self._data.next_states.append(next_state)
    self._data.rewards.append(reward)
    self._data.dones.append(done)

  def next_batch(self, batch_size):
    batch_indices = np.random.choice(len(self._data.states), batch_size)
    batch_states = np.array([self._data.states[i] for i in batch_indices])
    batch_actions = np.array([self._data.actions[i] for i in batch_indices])
    batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
    batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
    batch_dones = np.array([self._data.dones[i] for i in batch_indices])
    return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

def make_epsilon_greedy_policy(estimator, epsilon, nA):
  """
  Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
  
  Args:
      estimator: An estimator that returns q values for a given state
      epsilon: The probability to select a random action . float between 0 and 1.
      nA: Number of actions in the environment.
  
  Returns:
      A function that takes the observation as an argument and returns
      the probabilities for each action in the form of a numpy array of length nA.
  
  """
  def policy_fn(sess, observation):
    A = np.ones(nA, dtype=float) * epsilon / nA
    q_values = estimator.predict(sess, observation)
    best_action = np.argmax(q_values)
    A[best_action] += (1.0 - epsilon)
    return A
  return policy_fn

def q_learning(sess, env, approx, num_episodes, max_time_per_episode, target, discount_factor=0.99, epsilon=0.1, batch_size=128):
  """
  Q-Learning algorithm for off-policy TD control using Function Approximation.
  Finds the optimal greedy policy while following an epsilon-greedy policy.
  Implements the options of online learning or using experience replay and also
  target calculation by target networks, depending on the flags. You can reuse
  your Q-learning implementation of the last exercise.

  Args:
    env: OpenAI environment.
    approx: Action-Value function estimator
    num_episodes: Number of episodes to run for.
    max_time_per_episode: maximum number of time steps before episode is terminated
    discount_factor: gamma, discount factor of future rewards.
    epsilon: Chance to sample a random action. Float betwen 0 and 1.
    use_experience_replay: Indicator if experience replay should be used.
    batch_size: Number of samples per batch.
    target: Slowly updated target network to calculate the targets. Ignored if None.

  Returns:
    An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
  """

  # Keeps track of useful statistics
  stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))    
  replay_buffer = ReplayBuffer()
  for i_episode in range(num_episodes):
      
    # The policy we're following
    policy = make_epsilon_greedy_policy(
        approx, epsilon, env.action_space.n)
    
    # Print out which episode we're on, useful for debugging.
    # Also print reward for last episode
    last_reward = stats.episode_rewards[i_episode - 1]
    print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
    sys.stdout.flush()
    
    # Reset the environment and pick the first action
    state = env.reset()
    
    # One step in the environment
    for t in itertools.count():
      action_probs = policy(sess, [state])
      action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

      # Take a step
      next_state, reward, done, _ = env.step(action)

      stats.episode_rewards[i_episode] += reward
      stats.episode_lengths[i_episode] = t

      replay_buffer.add_transition(state, action, next_state, reward, done)
      batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = replay_buffer.next_batch(batch_size)
      double_q_actions = np.argmax(approx.predict(sess, batch_next_states), axis=1)
      q_values_next = target.predict(sess, batch_next_states)[np.arange(batch_size), double_q_actions]

      td_targets = batch_rewards.astype(np.float32)
      td_targets[np.logical_not(batch_dones)] += discount_factor * q_values_next[np.logical_not(batch_dones)]

      # Update the function approximator using our target
      loss = approx.update(sess, batch_states, batch_actions, td_targets)
      if target is not None:
        target.update(sess)
      if done or t >= max_time_per_episode:
        break
          
      state = next_state
  
  return stats

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
  # Plot the episode length over time
  fig1 = plt.figure(figsize=(10,5))
  plt.plot(stats.episode_lengths)
  plt.xlabel("Episode")
  plt.ylabel("Episode Length")
  plt.title("Episode Length over Time")
  fig1.savefig('episode_lengths.png')
  if noshow:
      plt.close(fig1)
  else:
      plt.show(fig1)

  # Plot the episode reward over time
  fig2 = plt.figure(figsize=(10,5))
  rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
  plt.plot(rewards_smoothed)
  plt.xlabel("Episode")
  plt.ylabel("Episode Reward (Smoothed)")
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
  fig2.savefig('reward.png')
  if noshow:
      plt.close(fig2)
  else:
      plt.show(fig2)

if __name__ == "__main__":
  env = MountainCarEnv()
  approx = NeuralNetwork()
  target = TargetNetwork()

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  stats = q_learning(sess, env, approx, 1000, 1000, target, batch_size=128)
  plot_episode_stats(stats)

  for _ in range(5):
    state = env.reset()
    for _ in range(1000):
      env.render()
      state,_,done,_ = env.step(np.argmax(approx.predict(sess, [state])))
      if done:
        break