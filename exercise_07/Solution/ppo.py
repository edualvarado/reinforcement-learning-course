import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
import itertools
import pandas as pd
from PIL import Image

from mountain_car import MountainCarEnv

"""
* -------------------------------------------------------------------------------
* There are TODOs in Policy Class!
* -------------------------------------------------------------------------------
"""

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
VALID_ACTIONS = [0, 1, 2]

# epsilon=0.2

class Policy():
  def __init__(self, name):
    self._name = name
    self._build_model(name)

  def _build_model(self, name):
    """
    Builds the Tensorflow graph.
    """

    self.states_pl = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    # The real return value
    self.targets_pl = tf.placeholder(shape=[None], dtype=tf.float32)
    # Integer id of which action was selected
    self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32)

    self.value_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    self.old_probabilites_pl = tf.placeholder(shape=[None, 3], dtype=tf.float32)

    batch_size = tf.shape(self.states_pl)[0]

    with tf.variable_scope(name):
      self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, 20, activation_fn=tf.nn.relu,
        weights_initializer=tf.random_uniform_initializer(0, 0.5))
      self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, activation_fn=tf.nn.relu,
        weights_initializer=tf.random_uniform_initializer(0, 0.5))
      self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 3, activation_fn=None,
        weights_initializer=tf.random_uniform_initializer(0, 0.5))
      
      self.predictions = tf.contrib.layers.softmax(self.fc3)

    gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
    self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
    self.old_action_predictions = tf.gather(tf.reshape(self.old_probabilites_pl, [-1]), gather_indices)

    ratio = self.action_predictions/self.old_action_predictions
    advantage = self.targets_pl-self.value_pl
    clipped_surrogate = tf.minimum(ratio * advantage, tf.clip_by_value(ratio, 0.9, 1.1)*advantage)

    self.objective = -tf.reduce_mean(clipped_surrogate)
    
    self.optimizer = tf.train.AdamOptimizer(0.0001)
    self.train_op = self.optimizer.minimize(self.objective)

  def predict(self, sess, s):
    """
    Args:
      sess: TensorFlow session
      states: array of states for which we want to predict the actions.
    Returns:
      The prediction of the output tensor.
    """
    probs = sess.run(self.predictions, { self.states_pl: s })
    actions = np.array([np.random.choice(VALID_ACTIONS, p=p) for p in probs])
    return actions, probs

  def update(self, sess, s, a, r, v, p):
    """
    Updates the weights of the neural network, based on its targets, its
    predictions, its loss and its optimizer.
    
    Args:
      sess: TensorFlow session.
      states: [current_state] or states of batch
      actions: [current_action] or actions of batch
      targets: [current_target] or targets of batch
    """
    feed_dict = { self.states_pl: s, self.targets_pl: r, self.actions_pl: a, self.value_pl: v, self.old_probabilites_pl: p}
    sess.run(self.train_op, feed_dict)

class OldPolicy(Policy):
  """
  Policy backup.
  """
  def __init__(self, name, associate_name):
    Policy.__init__(self, name)
    self._associate = self._register_associate(associate_name)

  def _register_associate(self, associate_name):
    own_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
    associate_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=associate_name)
    op_holder = [oldp.assign(p) for p, oldp in zip(associate_params, own_params)]
    return op_holder
      
  def update(self, sess):
    for op in self._associate:
      sess.run(op)

class ValueFunction():
  def __init__(self):
    self._build_model()

  def _build_model(self):
    self.states_pl = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    # The TD target value
    self.targets_pl = tf.placeholder(shape=[None], dtype=tf.float32)

    batch_size = tf.shape(self.states_pl)[0]

    self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, 20, weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.predictions = tf.contrib.layers.fully_connected(self.fc2, 1, activation_fn=None, weights_initializer=tf.random_uniform_initializer(0, 0.5))

    # Calcualte the loss
    self.losses = tf.squared_difference(self.targets_pl, self.predictions)
    self.loss = tf.reduce_mean(self.losses)

    self.optimizer = tf.train.AdamOptimizer(0.0001)
    self.train_op = self.optimizer.minimize(self.loss)
  
  def predict(self, sess, states):
    prediction = sess.run(self.predictions, { self.states_pl: states })
    return prediction

  def update(self, sess, states, targets):
    feed_dict = { self.states_pl: states, self.targets_pl: targets}
    loss = sess.run([self.train_op, self.loss], feed_dict)
    return loss

class ReplayBuffer:
  #Replay buffer for experience replay. Stores transitions.
  def __init__(self):
    self._data = namedtuple("ReplayBuffer", ["states", "actions", "returns"])
    self._data = self._data(states=[], actions=[], returns=[])

  def add_episode_data(self, states, actions, returns):
    self._data.states.extend(states)
    self._data.actions.extend(actions)
    self._data.returns.extend(returns)

  def next_batch(self, batch_size):
    batch_indices = np.random.choice(len(self._data.states), batch_size)
    batch_states = np.array([self._data.states[i] for i in batch_indices])
    batch_actions = np.array([self._data.actions[i] for i in batch_indices])
    batch_returns = np.array([self._data.returns[i] for i in batch_indices])
    return batch_states, batch_actions, batch_returns

def ppo(sess, env, policy, old_policy, value_function, num_episodes, discount_factor=1.0, update_frequency=1, gradient_steps=15, batch_size=32):
  stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes)) 
  old_policy.update(sess)
  replay_buffer = ReplayBuffer()
  for i_episode in range(1, num_episodes + 1):
    # Generate an episode.
    # An episode is an array of (state, action, reward) tuples
    states = []
    actions = []
    rewards = []
    state = env.reset()
    for t in range(500):
      #action, _ = policy.predict(sess, [state])
      action, _ = old_policy.predict(sess, [state])
      action = action[0]
      next_state, reward, done, info = env.step(action)

      stats.episode_rewards[i_episode-1] += reward
      stats.episode_lengths[i_episode-1] = t
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      if done:
        break
      state = next_state

    returns = [sum([r*(discount_factor**i) for i,r in enumerate(rewards[t:])]) for t in range(len(rewards))]
    replay_buffer.add_episode_data(states, actions, returns)

    if i_episode % update_frequency == 0:
      #update stuff
      for _ in range(gradient_steps):
        batch_states, batch_actions, batch_returns = replay_buffer.next_batch(batch_size)
        value_function.update(sess, batch_states, batch_returns)
        predicted_state_values = value_function.predict(sess, batch_states)
        old_probabilities = old_policy.predict(sess, batch_states)[1]
        policy.update(sess, batch_states, batch_actions, batch_returns, predicted_state_values, old_probabilities)
      old_policy.update(sess)
      replay_buffer = ReplayBuffer()
      
    print("\r{} Steps in Episode {}/{}. Reward {}".format(len(rewards), i_episode, num_episodes, sum([r for i,r in enumerate(rewards)])))
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
  tf.reset_default_graph()
  env = MountainCarEnv()
  pi = Policy("Policy")
  old_pi = OldPolicy("OldPolicy", "Policy")
  v = ValueFunction()

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  stats = ppo(sess, env, pi, old_pi, v, 3000)

  plot_episode_stats(stats)

  for _ in range(5):
    state = env.reset()
    for i in range(500):
      env.render()
      _, _, done, _ = env.step(pi.predict(sess, [state])[0][0])
      if done:
        break