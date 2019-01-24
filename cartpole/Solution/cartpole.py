import gym
import sys

import numpy as np
import matplotlib.pyplot as plt

from continuous_cartpole import ContinuousCartPoleEnv

# Create the Cart-Pole game environment
env = ContinuousCartPoleEnv()

rewards_list = []
steps_list = []
num_episodes = 5
episodes_list = np.arange(1,num_episodes + 1)

# Number of episodes
for i_episode in range(num_episodes):
    print ("")
    print ("========= EPISODE %d =========" % (i_episode + 1))
    observation = env.reset()
    total_reward = 0
    
    # Number of time-steps
    for t in range(100):
        env.render()
        action = env.action_space.sample() # Take random action
        observation, reward, done, info = env.step(action)
        total_reward += reward
        '''
        print("----------- Begin time-step %d ----------" % (t))
        print("Observation (4-array): ", observation)
        print("Action (0-left or 1-right): ", action)        
        print("Reward: ", reward)
        print("Is it done?", done)
        print("-----------------------------------------")
        '''
        if t == 99:
            print("Time is over")
            rewards_list.append(total_reward)
            steps_list.append(t)
        # When is done, executes
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            rewards_list.append(total_reward)
            steps_list.append(t)
            break            
    print ("Total reward in this episode: ", total_reward)
    
print ("")
print ("========== RESULTS ==========")
print ("=============================")
print ("Total number of episodes: ", num_episodes)
print ("Average reward of all episodes: ", np.average(rewards_list))
print ("")

env.close()
        

############################
# Plots
############################

'''
# summarize history for loss
fig1 = plt.figure()
plt.plot(loss_list)
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('steps')
#plt.show()
fig1.savefig('model_loss.png')
'''

# summarize history for loss
fig2 = plt.figure()
plt.plot(episodes_list, rewards_list)
plt.title('Total Reward')
plt.ylabel('Total Reward')
plt.xlabel('Episode')
#plt.show()
fig2.savefig('model_reward.png')

# summarize history for loss
fig3 = plt.figure()
plt.plot(episodes_list, steps_list)
plt.title('Total Steps')
plt.ylabel('Total Steps')
plt.xlabel('Episode')
#plt.show()
fig3.savefig('model_steps.png')

'''
# summarize history for loss
fig4 = plt.figure()
plt.plot(epsilon_list)
plt.title('Epsilon Decay`')
plt.ylabel('Epsilon')
plt.xlabel('Total Steps')
#plt.show()
fig4.savefig('epsilon_list.png')
'''

#np.savetxt("history_loss.txt", loss_list, delimiter=",", fmt="%.5f")
np.savetxt("total_reward.txt", rewards_list, delimiter=",", fmt="%.5f")
np.savetxt("total_steps.txt", steps_list, delimiter=",", fmt="%.5f")
#np.savetxt("epsilon_list.txt", epsilon_list, delimiter=",", fmt="%.5f")
