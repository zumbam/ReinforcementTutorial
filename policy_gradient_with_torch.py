import gym
import numpy as np
import os
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt




class PolicyNet(torch.nn.Module):
    """
    This is the policy network. It estimates a distribution for choosing a appropriate action for a given state.
    """

    def __init__(self):
        super(PolicyNet, self).__init__()
        self.lin_1 = torch.nn.Linear(4, 10)
        self.softsign = torch.nn.Softsign()
        # self.lin_2 = torch.nn.Linear(10, 10)
        self.lin_3 = torch.nn.Linear(10, 2)
        # self.softmax = torch.nn.Softmax()

    def forward(self, *input, **kwargs):
        observation_ = input[0]
        net = self.lin_1(observation_)
        net = self.softsign(net)
        # net = self.lin_2(net)
        # net = self.softsign(net)
        net = self.lin_3(net)
        # net = self.softmax(net)
        return net




def sample_trajectories(model, num_samples, max_num_sequence_elems=10):
    '''
    "On Policy sampling" for preparing trajectories for a given model.

    :param model: The policy model
    :param num_samples: the batchsize (or number of samples for monte carlo estimation)
    :param max_num_sequence_elems: the maximum number of elements till game is done (default in cartpole is 200)
    :return: list for all sample trajectories containing [probabilities for each action decision, given rewards,
     the action taken
    '''
    softmax = torch.nn.Softmax()

    # sampling random numbers for all possible states and samples.
    sampling = np.random.rand(num_samples, max_num_sequence_elems)

    actions_list = []
    rewards_list = []
    observations_list = []
    probs_list = []

    for i in range(num_samples):
        state = env.reset()
        actions = []
        rewards = []
        probs = []
        observations = []
        for j in range(max_num_sequence_elems):

            # evaluating the policy network for action probabilities.
            observation_tensor = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32)
            res = model(observation_tensor)

            prob = softmax(res)[0]
            action = 0
            reward = 0
            # making discrete action sampling to possible explore the whole sampling space.
            if sampling[i, j] > prob[0]:
                action = 1
            else:
                action = 0

            probs.append(res)
            actions.append(action)
            observation, reward, done, info = env.step(action)

            observations.append(observation)
            rewards.append(reward)
            state = observation

            if done:
                break

        actions_list.append(actions)
        rewards_list.append(rewards)
        probs_list.append(probs)

    return probs_list, rewards_list, actions_list


def maximum_expected_reward_loss(model_, probs_, rewards_, actions_, loss_):
    '''
    This is function returns the policy loss function resulting in the policy gradient. Interestingly the cross
    entropy function loss has exactly the need attributes for the policy gradient "1/n * sum_b_B(sum_t_T (grad (log (pi(a|s)))) *G_T)".
    The cross entropy results exactly in grad (log (pi(a|s)).

    :param model_: the policy model
    :param probs_: the probabilities.: tensor [[a, b], [a, b]]
    :param rewards_: the rewards collected a trajectory.: tensor [r1, r2, ..., rn]
    :param actions_: the actions taken in the samples.: tensor [a1, a2, ..., an]
    :param loss_: the loss function to use. (avoid recreation of graph nodes).: torch.nn.loss
    :return: the mean maximum likelihood loss over all samples.: tensor
    '''

    # this is the used discount function decaying till for giving long time goals less attention.
    decaying_discount_factor = torch.tensor(np.linspace(1, 0.1, max_num_sequence_elems))
    losses = []

    for b in range(len(probs_)):
        rewards = rewards_[b]
        probs = probs_[b]
        actions = actions_[b]
        discounted_reward_args = [decaying_discount_factor[i] * rewards[i] for i in range(len(rewards))]
        G = torch.sum(torch.stack(discounted_reward_args))

        log_probs_args = torch.stack([probs[i] for i in range(len(probs))]).squeeze(1)
        actions_tensor = torch.tensor(actions)
        log_probs = loss_(log_probs_args, actions_tensor)
        losses.append(log_probs * G)

    loss_tensor = torch.stack(losses)
    loss = -torch.mean(loss_tensor)
    return loss

def measure_expected_reward(rewards_):
    '''
    This function evaluates the current mean expectation.
    :param rewards_: the reward from a sampled trajectory
    :return: the mean reward over all given trajectories.:scalar
    '''
    reward_sums = []
    for b in range(len(rewards_)):
        rewards = np.array(rewards_[b])
        reward_sum = np.sum(rewards)
        reward_sums.append(reward_sum)
    mean_reward = np.mean(np.array(reward_sums))
    return mean_reward

# num of experiment to run
num_episodes = 3000
# num of samples to use for monte carlo estimation
batch_size = 32
# maximum number of states till game is done
max_num_sequence_elems = 200

# init playground
env = gym.make('CartPole-v1')
# init network
policy_net = PolicyNet()

# init optimization needs
optimizer = torch.optim.SGD(policy_net.parameters(), lr=0.01)
expected_rewards = []
steps = []
entropy = torch.nn.CrossEntropyLoss()
for i in range(num_episodes):

    probs, rewards, actions = sample_trajectories(policy_net, batch_size, max_num_sequence_elems=max_num_sequence_elems)
    loss = maximum_expected_reward_loss(policy_net, probs, rewards, actions, entropy)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    if i % 10 == 0:
        print(f'reach episode {i}')
        print(f'loss {loss}')
        expected_reward = measure_expected_reward(rewards)
        print(f'expected reward {expected_reward}')
        steps.append(i)
        expected_rewards.append(expected_reward)
    # if i % 100 == 0:

    if i % 1000 == 0:
        d = {'steps': steps, 'expected_reward': expected_rewards}
        df = pd.DataFrame(data=d)
        df.to_csv(f'expected_reward{i}.csv')
        torch.save(policy_net.state_dict(), f'mynet_{i}.pt')

d = {'steps': steps, 'expected_reward': expected_rewards}
df = pd.DataFrame(data=d)
df.to_csv('expected_reward.csv')


sns.set()
sns.relplot(x='steps', y='expected_reward', data=df, kind='line')

torch.save(policy_net.state_dict(), 'mynet_last.pt')
plt.show()

print(f'reach episode {num_episodes}')