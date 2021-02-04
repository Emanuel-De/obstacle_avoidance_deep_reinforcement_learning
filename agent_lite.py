# coding=utf-8

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # code whether or not to run on the graphics card;
# if active then only cpu, if commented out then run on gpu
import environment_lite as environment
import time as t
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

env = environment.cls_Environment()  # create environment object


# function for the discounted reward
def calc_discounted_rewards(reward, gamma):
    discounted_reward = np.zeros_like(reward)
    reward_add = 0
    for t in reversed(range(0, reward.size)):
        reward_add = reward_add * gamma + reward[t]
        discounted_reward[t] = reward_add
    return discounted_reward


# function to print information on step
def print_data_fcn(most_frequent_action, current_reward, total_steps, cur_ep, done_info,
                   win_count, lose_count, reason, state_next, loss, crash_reason_count):
    print('most common action: ' + str(most_frequent_action))
    print('most common crashes: ' + str(crash_reason_count))
    print('current reward: ' + str(current_reward), 'loss: ' + str(loss))
    print('steps / ep: ' + str(total_steps))
    print('episode number: ' + str(cur_ep))
    print('speed: ' + str(state_next[16]), 'steering angle: ' + str(state_next[17]))
    if done_info == -1:
        print('crashed: ' + str(reason))
    elif done_info == 1:
        print('WIN!')
    print('time episode: ' + str(t.clock() - t_ep))
    print_reward = np.mean(total_reward[-50:])
    print('mean total reward last 50 ep: ' + str(print_reward))
    print('win: ' + str(win_count), 'lose: ' + str(lose_count), '\n')


# defining some variables for bookkeeping
total_reward = []  # holder for the reward for every eipsode during a training session (number of traing episodes)
mean_reward = []  # holder for the mean reward over a number of defined past episodes
print_plot = 0  # counter to limit the number of plots printed
win_count = 0  # counter to count successful episodes
lose_count = 0  # counter to count failed episodes
win_count_holder = []  # holder to plot the development of successful episodes
lose_count_holder = []  # holder to plot the development of failed episodes
crash_reason_count_holder_total = []  # holder for the reason the agent crashed, to count how many times,
# which done criteria was triggered
t0 = t.clock()  # current time

# functional variables
cur_ep = 0  # counter for the episodes
total_nr_training_ep = 5000  # nr of training loops

# control variables for saving and loading the model
load_model = True  # Whether to load a saved model or not
path = "./dqn"  # path to where the model is saved

# some tf housekeeping
tf.reset_default_graph()  # reset the graph

# definition of hyperparameters
gamma = 0.99  # discount rate gamma
learning_rate = 1e-5  # learning rate for the network / optimizer
update_frequency = 3  # nr of episodes before the network is updated (mini batches)

# definition of neural network
D_input = 19  # number of input neurons
# number of neurons per hidden layer
H_w1_w2 = 80
H_w2_w3 = 80
H_w3_w4 = 80
H_w4_w5 = 40
D_output = 3  # number of output parameters, size of action space

# W matrix with size shape are the weights
# b vector with size shape are the bias
# layer matrix with size shape the layers connecting the weights with the activation fcn
input_vector = tf.placeholder(dtype=tf.float32, shape=[None, D_input])  # input vector as placeholder variable
W1 = tf.get_variable(name="W1", shape=[D_input, H_w1_w2], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable(name="b1", shape=[H_w1_w2], initializer=tf.ones_initializer())
layer1 = tf.nn.leaky_relu(tf.matmul(input_vector, W1) + b1)
W2 = tf.get_variable(name="W2", shape=[H_w1_w2, H_w2_w3], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable(name="b2", shape=[H_w2_w3], initializer=tf.ones_initializer())
layer2 = tf.nn.leaky_relu(tf.matmul(layer1, W2) + b2)
W3 = tf.get_variable(name="W3", shape=[H_w2_w3, H_w3_w4], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable(name="b3", shape=[H_w3_w4], initializer=tf.ones_initializer())
layer3 = tf.nn.leaky_relu(tf.matmul(layer2, W3) + b3)
W4 = tf.get_variable(name="W4", shape=[H_w3_w4, H_w4_w5], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable(name="b4", shape=[H_w4_w5], initializer=tf.ones_initializer())
layer4 = tf.nn.leaky_relu(tf.matmul(layer3, W4) + b4)
W5 = tf.get_variable(name="W5", shape=[H_w4_w5, D_output], initializer=tf.contrib.layers.xavier_initializer())
output_vector = tf.nn.softmax(tf.matmul(layer4, W5))  # output vector giving the probabilities of the actions

# next section to determine the gradient by how much the weights have to be adjusted per episode
# this is done after each episode is finished
advantage = tf.placeholder(shape=[None], dtype=tf.float32)  # advantage is a placeholder vector for the result of the
# reward per step after running the reward vector through the discounted reward function (discounted reward)

actions_taken = tf.placeholder(shape=[None], dtype=tf.int32)  # actions_taken is a placeholder for the actions_taken
# taken by the agent as the index of the highest probability of the output vector (result of argmax function)

Number_steps = tf.shape(output_vector)[0]  # Number_steps is a scalar to hold the number of steps the episode had
# this is done after running all the states of the episode through the neural net again obtaining the probability for
# for each possible action for each state
D_actionspace = tf.shape(output_vector)[1]  # D_actionspace is a scalar to hold the size of the action space

flat_output_vector = tf.reshape(output_vector, [-1])  # flat_output_vector is a vector to hold the reshaped
# output vector from a [number of steps, number of actions_taken] matrix to a
# [1, number of steps * number of actions_taken] vector shape. The flat_output_vector now contains all probabilities
#  in one line vector
index_of_taken_action = tf.range(0, Number_steps) * D_actionspace + actions_taken  # index_of_taken_action: determine
# the column of the probability for the actions chosen in the flat_output_vector
prob_actions_taken = tf.gather(flat_output_vector, index_of_taken_action)  # prob_actions_taken holds the probabilities
# of the chosen actions

loss_fcn = -tf.reduce_mean(tf.log(prob_actions_taken) * advantage)  # loss_fcn calculates the loss value
# this is done as the mean value of the log of the probability of the action multiplied by the discounted reward
# for each step

trainable_variables = tf.trainable_variables()  # tf holder for all the trainable variables

episode_gradient = tf.gradients(loss_fcn, trainable_variables)  # episode_gradient: holds the gradients for how the
# weights (trainable variables) have to be adjusted


# following section of code is to apply the calculated gradients to the weights i.e. trainable variables
episode_gradients = []
for index, t_var in enumerate(trainable_variables):  # create tensorflow placeholders for the gradients and append them to a holder
    placeholder = tf.placeholder(dtype=tf.float32, name=str(index) + '_holder')
    episode_gradients.append(placeholder)

trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # define the tf back propagation method
nn_update = trainer.apply_gradients(zip(episode_gradients, trainable_variables))  # apply the gradients to
# the trainable variables


# more tf housekeeping: set up initializer and saver
init = tf.global_variables_initializer()  # set variable initializer
saver = tf.train.Saver()  # set the saver to save model

if not os.path.exists(path):  # Make sure the path to save the model exists
    os.makedirs(path)

# next section sets up tf session and runs the training
with tf.Session() as sess:
    sess.run(init)  # initializer sets up the layers and variables (computational graph)

    if load_model is True:  # load model if variable is set True
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    gradient_Buffer = sess.run(tf.trainable_variables())  # set up the gradient buffer that contains the trainable
    #  variables this way tensorflow knows which variable to update with which gradient
    for index, gradient in enumerate(gradient_Buffer):  # make sure the gradient buffer is empty by setting everything 0
        gradient_Buffer[index] = gradient * 0

    while cur_ep < total_nr_training_ep:  # run the code until the number of training episodes is reached

        state, pos_agent, dist_vec, runtime = env.reset()  # get the first and initial state from the environment.
        # Other variables are for debug and control purposes.

        current_reward = 0  # counter for the current reward, for bookkeeping purposes
        current_episode = []  # holder to hold the observations (state, action, reward) for an episode
        total_steps = 0  # counter for the number of steps taken in one episode

        t_ep = t.clock()  # get current time to calculate the time of the episode

        while True:

            nn_state = np.float32(state)
            act_dist = sess.run(output_vector, feed_dict={input_vector: [nn_state]})  # feed the
            #  state vector to the network and obtain the probabilities of the different actions

            actions = np.linspace(0, (D_output-1), D_output, dtype=int)  # create a vector with the actions
            action_send = np.random.choice(a=actions, p=act_dist[0])  # chose an action with the probability
            # distribution of the action vector itself

            state_next, reward, done, pos_agent, dist_vec, runtime, done_info, reason = env.step(action_send)  # hand
            # the chosen action to the environment and get the next state, the reward and information if terminated
            # or not (done). Other variables are for debug and control purposes.

            action_send = np.float32(action_send)
            reward = np.float32(reward)

            current_episode.append([nn_state, action_send, reward])  # save the observation

            state = state_next  # set the received state from the environment as current state to run though NN on next
            # iteration

            current_reward += reward  # bookkeeping variable reward for episode. Same as sum over current_episode[:, 2]
            # before discount fcn

            total_steps += 1  # increment number of steps for bookkeeping purposes

            if done:  # if episode is terminated then run this code

                crash_reason_count_holder_total.append(reason)  # save reason for episode termination for bookkeeping

                current_episode = np.array(current_episode)  # turn observation holder into np array
                current_episode[:, 2] = calc_discounted_rewards(current_episode[:, 2], gamma)  # run the rewards
                # returned by the environment through the discounted reward function

                feed_dict = {advantage: current_episode[:, 2], actions_taken: current_episode[:, 1],
                             input_vector: np.vstack(current_episode[:, 0])}  # dictionary to feed to the NN to

                # calculate gradients. Other variables are used for debug purposes and have no functional use
                gradients, act_taken, idx_of_actions, flatoutput, outputvec, loss = sess.run([episode_gradient,
                                                                                              prob_actions_taken,
                                                                                              index_of_taken_action,
                                                                                              flat_output_vector,
                                                                                              output_vector,
                                                                                              loss_fcn],
                                                                                             feed_dict=feed_dict)
                # run the values through the NN and training algorithm to obtain gradients. Other variables are for
                # Other variables are for debug and control purposes.

                for index, gradient in enumerate(gradients):  # add all the new calculated gradients to the previously
                    # calculated gradients
                    gradient_Buffer[index] += gradient

                if cur_ep % update_frequency == 0 and cur_ep != 0:  # run this code if the update frequency is reached
                    #  run this code to apply the gradients
                    feed_dict_update = dict(zip(episode_gradients, gradient_Buffer))  # create the dictionary to feed
                    # to the gradient apply method
                    _ = sess.run([nn_update], feed_dict=feed_dict_update)  # Apply the gradients. No return value
                    # because its only relevant that the gradients are applied

                    for index, gradient in enumerate(gradient_Buffer):  # make sure the gradient buffer is empty
                        # by setting everything 0
                        gradient_Buffer[index] = gradient * 0

                total_reward.append(current_reward)  # append reward for bookkeeping
                break  # break to go to new episode

        # following code is for bookkeeping, plotting and debug and to see how well agent is doing
        most_frequent_action = Counter(current_episode[:, 1]).most_common(5)  # bookkeeping most frequent action
        crash_reason_count_total = Counter(crash_reason_count_holder_total).most_common(7)  # bookkeeping most
        # frequent done criteria

        # count wins and losses
        if done_info == -1:
            lose_count += 1
        elif done_info == 1:
            win_count += 1

        # calculate mean reward after defined number of episodes
        if cur_ep % 50 == 0:
            mean_reward.append(np.mean(total_reward[-50:]))
            win_count_holder.append(win_count)
            lose_count_holder.append(lose_count)

        # when to plot and print information
        if (done_info == 1) and (print_plot < 1):
            env.render()
            print_data_fcn(most_frequent_action, current_reward, total_steps, cur_ep, done_info, win_count, lose_count,
                           reason, state_next, loss, crash_reason_count_total)
            print_plot += 1
        elif (cur_ep % 200 == 0) or (cur_ep <= 30) or (cur_ep >= total_nr_training_ep - 10):
            env.render()
            print_data_fcn(most_frequent_action, current_reward, total_steps, cur_ep, done_info, win_count, lose_count,
                           reason, state_next, loss, crash_reason_count_total)

        cur_ep += 1  # increment episode number

    saver.save(sess, path + '/model-' + str(t0) + '.ckpt')  # save model
    print("Saved Model \n")

# print and plot some more stats at and of training
print('time total [min]: ' + str((t.clock() - t0) / 60))
print('time total [sec]: ' + str(t.clock() - t0))
print('win: ' + str(win_count), 'lose: ' + str(lose_count))
print('training complete')
plt.plot(mean_reward, 'b')
plt.xlabel('Mean number of episodes')
plt.ylabel('Mean Reward')
plt.show()
plt.plot(win_count_holder, 'g')
plt.plot(lose_count_holder, 'r')
plt.xlabel('Mean number of episodes')
plt.ylabel('Number of wins and losses')
plt.show()
