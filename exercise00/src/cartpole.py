import gym

# CartPole from gym
env = gym.make('CartPole-v0')

# Number of episodes
for i_episode in range(20):
    print ("")
    print ("========= EPISODE %d =========" % (i_episode))
    observation = env.reset()
    # Number of time-steps
    for t in range(100):
        env.render()
        action = env.action_space.sample() # Take random action
        observation, reward, done, info = env.step(action)
        print("----------- Begin time-step %d ----------" % (t))
        print("Observation (4-array): ", observation)
        print("Action (0-left or 1-right): ", action)        
        print("Reward: ", reward)
        print("Is it done?", done)
        print("-----------------------------------------")
        # When is done, executes
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


""" Basic algorithm to load env
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
"""