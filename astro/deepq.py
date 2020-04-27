import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQ(nn.Module):
    def __init__(self, lr, input_dim, fc1_dim, fc2_dim, n_actions):
        super(DeepQ, self).__init__()

        self.input_dim = input_dim

        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, obs):
        state = T.tensor(obs)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        actions = self.fc3(x)

        return actions

class Agent(object):
    def __init__(self, gamma, epsilon, lr, input_dim, batch_size, \
                n_actions, max_mem_size=1000000, eps_min=0.1, eps_dec=0.996):

        self.gamma        = gamma
        self.epsilon      = epsilon
        self.eps_dec      = eps_dec
        self.eps_min      = eps_min
        self.batch_size   = batch_size
        self.n_actions    = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size     = max_mem_size

        self.mem_counter = 0

        self.Q_eval = DeepQ(lr, input_dim=input_dim, fc1_dim=256, fc2_dim=256, n_actions=n_actions)

        self.state_mem     = np.zeros((self.mem_size, *input_dim))
        self.new_state_mem = np.zeros((self.mem_size, *input_dim))
        self.action_mem    = np.zeros((self.mem_size, self.n_actions), dtype=np.uint8)
        self.reward_mem    = np.zeros(self.mem_size)
        self.terminal_mem  = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_counter % self.mem_size

        self.state_mem[index] = state

        # one-hot encoding
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0

        self.action_mem[index]    = actions
        self.reward_mem[index]    = reward
        self.terminal_mem[index]  = terminal
        self.new_state_mem[index] = state_

    def choose_action(self, obs):
        rand = np.random.random()

        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.Q_eval.forward(obs)
            action  = T.argmax(actions).item()

        return action

    def learn(self):
        if self.mem_counter > self.batch_size:
            self.Q_eval.optimizer.zero_grad()

            if self.mem_counter < self.mem_size:
                max_mem = self.mem_counter
            else:
                max_mem = self.mem_size

            batch = np.random.choice(max_mem, self.batch_size)

            state_batch     = self.state_mem[batch]
            action_batch    = self.action_mem[batch]
            action_values   = np.array(self.action_space, dtype=np.uint8)
            action_indices  = np.dot(action_batch, action_values)
            reward_batch    = self.reward_mem[batch]
            terminal_batch  = self.terminal_mem[batch]
            new_state_batch = self.new_state_mem[batch]

            q_eval   = self.Q_eval.forward(state_batch)
            q_target = q_eval.clone()

            q_next   = self.Q_eval.forward(new_state_batch)

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward_batch + self.gamma * T.max(q_next, dim=1)[0] * terminal_batch

            if self.epsilon > self.eps_min:
                self.epsilon = self.epsilon * self.eps_dec
            else:
                self.epsilon = self.eps_min

            loss = self.Q_eval.loss(q_target, q_eval)
            loss.backward()

            self.Q_eval.optimizer.step()