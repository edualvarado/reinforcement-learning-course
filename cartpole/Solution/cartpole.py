import gym
import sys

from continuous_cartpole import ContinuousCartPoleEnv

# Create the Cart-Pole game environment
env = ContinuousCartPoleEnv()

# Number of episodes
for i_episode in range(20):
    print ("")
    print ("========= EPISODE %d =========" % (i_episode))
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
        # When is done, executes
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        if t == 99:
            print("Time is over")
            
    print ("Total reward: ", total_reward)
    
env.close()
        

############################
# Plots
############################

# summarize history for loss
fig1 = plt.figure()
plt.plot(loss_list)
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('steps')
#plt.show()
fig1.savefig('model_loss.png')

# summarize history for loss
fig2 = plt.figure()
plt.plot(rewards_list)
plt.title('Total Reward')
plt.ylabel('Episode')
plt.xlabel('Total Reward')
#plt.show()
fig2.savefig('model_reward.png')

# summarize history for loss
fig3 = plt.figure()
plt.plot(steps_list)
plt.title('Total Steps')
plt.ylabel('Episode')
plt.xlabel('Total Steps')
#plt.show()
fig3.savefig('model_steps.png')

# summarize history for loss
fig4 = plt.figure()
plt.plot(epsilon_list)
plt.title('Epsilon Decay`')
plt.ylabel('Epsilon')
plt.xlabel('Total Steps')
#plt.show()
fig4.savefig('epsilon_list.png')

np.savetxt("history_loss.txt", loss_list, delimiter=",", fmt="%.5f")
np.savetxt("total_reward.txt", rewards_list, delimiter=",", fmt="%.5f")
np.savetxt("total_steps.txt", steps_list, delimiter=",", fmt="%.5f")
np.savetxt("epsilon_list.txt", epsilon_list, delimiter=",", fmt="%.5f")
