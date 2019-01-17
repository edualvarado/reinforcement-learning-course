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

class Policy():
  def __init__(self):
    self._build_model()

  def _build_model(self):
    """
    Builds the Tensorflow graph.
    """

    self.states_pl = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    # The TD target value
    self.targets_pl = tf.placeholder(shape=[None], dtype=tf.float32)
    # Integer id of which action was selected
    self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32)

    self.baseline_pl = tf.placeholder(shape=[None, None, None], dtype=tf.float32)

    batch_size = tf.shape(self.states_pl)[0]

    self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, 20, activation_fn=tf.nn.relu,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, activation_fn=tf.nn.relu,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 3, activation_fn=None,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))
    
    # -----------------------------------------------------------------------
    # TODO: Implement softmax output
    # -----------------------------------------------------------------------
    self.predictions = tf.contrib.layers.softmax(self.fc3)

    # Get the predictions for the chosen actions only
    gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
    self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

    # -----------------------------------------------------------------------
    # TODO: Implement the policy gradient objective. Do not forget to negate
    # -----------------------------------------------------------------------
    # the objective, since the predefined optimizers only minimize in
    # tensorflow.
    self.objective = -tf.reduce_mean(tf.log(self.action_predictions)*(self.targets_pl-self.baseline_pl))
    
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
    p = sess.run(self.predictions, { self.states_pl: s })[0]
    return np.random.choice(VALID_ACTIONS, p=p), p

  def update(self, sess, s, a, y, b):
    """
    Updates the weights of the neural network, based on its targets, its
    predictions, its loss and its optimizer.
    
    Args:
      sess: TensorFlow session.
      states: [current_state] or states of batch
      actions: [current_action] or actions of batch
      targets: [current_target] or targets of batch
    """
    feed_dict = { self.states_pl: s, self.targets_pl: y, self.actions_pl: a, self.baseline_pl: b }
    sess.run(self.train_op, feed_dict)

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

def reinforce(sess, env, policy, baseline, num_episodes, discount_factor=1.0):
  stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes)) 

  for i_episode in range(1, num_episodes + 1):
    # Generate an episode.
    # An episode is an array of (state, action, reward) tuples
    episode = []
    state = env.reset()
    for t in range(500):
      #if i_episode % 100 == 0:
      #  env.render()
      action, p = policy.predict(sess, [state])
      next_state, reward, done, info = env.step(action)

      stats.episode_rewards[i_episode-1] += reward
      stats.episode_lengths[i_episode-1] = t
      episode.append((state, action, reward))
      if done:
        break
      state = next_state

    for t in range(len(episode)):
      # Find the first occurance of the state in the episode
      state, action, reward = episode[t]
      # Sum up all rewards since the first occurance
      G = sum([e[2]*(discount_factor**i) for i,e in enumerate(episode[t:])])
      # Calculate average return for this state over all sampled episodes
      baseline.update(sess, [state], [G])
      policy.update(sess, [state], [action], [G], [baseline.predict(sess, [state])])
    print("\r{} Steps in Episode {}/{}. Reward {}".format(len(episode), i_episode, num_episodes, sum([e[2] for i,e in enumerate(episode)])))
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
  p = Policy()
  b = ValueFunction()

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  stats = reinforce(sess, env, p, b, 3000)

  plot_episode_stats(stats)
  saver = tf.train.Saver()
  saver.save(sess, "./policies.ckpt")

  for _ in range(5):
    state = env.reset()
    for i in range(500):
      env.render()
      _, _, done, _ = env.step(p.predict(sess, [state])[0])
      if done:
        break