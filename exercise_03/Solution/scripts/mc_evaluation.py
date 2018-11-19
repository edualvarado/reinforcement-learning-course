from collections import defaultdict
import sys

def mc_evaluation(policy, env, num_episodes, discount_factor=1.0):
  """
  First-visit MC Policy Evaluation. Calculates the value function
  for a given policy using sampling.

  Args:
    policy: A function that maps an observation to action probabilities.
    env: OpenAI gym environment.
    num_episodes: Nubmer of episodes to sample.
    discount_factor: Lambda discount factor.

  Returns:
    A dictionary that maps from state to value.
    The state is a tuple -- containing the players current sum, the dealer's one showing card (1-10 where 1 is ace) 
    and whether or not the player holds a usable ace (0 or 1) -- and the value is a float.
  """

  # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
  returns_sum = defaultdict(float)
  returns_count = defaultdict(float)
    
  # The final value function
  V = defaultdict(float)
  
  for i_episode in range(1, num_episodes + 1):
    # Print out which episode we're on, useful for debugging.
    if i_episode % 1000 == 0:
      print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
      sys.stdout.flush()

    # Generate an episode.
    # An episode is an array of (state, action, reward) tuples
    episode = []
    state = env.reset()
    for t in range(100):
      action = policy(state)
      next_state, reward, done, _ = env.step(action)
      episode.append((state, action, reward))
      if done:
        break
      state = next_state

    # Find all states the we've visited in this episode
    # We convert each state to a tuple so that we can use it as a dict key
    states_in_episode = set([tuple(x[0]) for x in episode])
    for state in states_in_episode:
      # Find the first occurance of the state in the episode
      first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
      # Sum up all rewards since the first occurance
      G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
      # Calculate average return for this state over all sampled episodes
      returns_sum[state] += G
      returns_count[state] += 1.0
      V[state] = returns_sum[state] / returns_count[state]

  return V