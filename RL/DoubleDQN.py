#DoubleDQN
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_ddqn(env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def to_tensor(x):
      return torch.from_numpy(x.astype(np.float32)).to(device)
    def to_tensor_long(x):
      return torch.tensor([[x]], device=device, dtype=torch.long)

    class Q_Network(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__()
            layers = [
                  nn.Linear(input_size, hidden_size), #1
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size), #2
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size), #3
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size), #4
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size), #5
                  nn.Tanh(),
                  nn.Linear(hidden_size, output_size) #6
            ]
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            q_values = self.model(x)
            return q_values

    Q = Q_Network(env.history_t * 7 +5, 200, 9).to(device)
    Q_ast = copy.deepcopy(Q).to(device)

    optimizer = optim.Adam(Q.parameters(), lr = 0.0001)
    loss_fn = nn.MSELoss()
    epoch_num = 300
    # epoch_num = 10
    step_max = len(env.data)-1
    memory_size = 10000
    batch_size = 100
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    # start_reduce_epsilon = 200
    train_freq = 10
    update_q_freq = 100
    gamma = 0.95
    show_log_freq = 5
    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []
    qvalue_ave = []##q_value
    reward_ave = []##reward
    profit_ave = []

    start = time.time()
    np.random.seed(42)
    for epoch in range(epoch_num):
        pobs = env.reset()
        step = 0
        done = False

        total_reward = 0
        total_loss = 0
        qvalue_memory = []##q_value
        reward_memory = []

        while not done and step < step_max:
            # select act
            pact = np.random.randint(9)

            if np.random.rand() > epsilon:
                input_state = np.array(pobs, dtype=np.float32).copy()
                # pact = Q(to_tensor(input_state))
                with torch.no_grad():
                  pact = Q(to_tensor(input_state))
                qvalue_memory.append(max(pact.detach().cpu().numpy()))
                pact = np.argmax(pact.detach().cpu().numpy().reshape(1, -1))
                # print(pact)

            # act
            obs, reward, done, pact = env.step(pact)
            reward_memory.append(reward)
            if pact == 9:
              pact = 5

            # add memory
            memory.append((pobs, pact, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)

            # train or update q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    shuffled_memory = np.random.permutation(memory)
                    memory_idx = range(len(shuffled_memory))

                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.float32)
                        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)
                        q = Q(to_tensor(b_pobs))
                        q_detach = q.detach().cpu().numpy()
                        indices = np.argmax(q_detach, axis=1)
                        maxqs = Q_ast(to_tensor(b_obs)).detach().cpu().numpy()
                        target = copy.deepcopy(q_detach)
                        for j in range(batch_size):
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxqs[j, indices[j]]*(not b_done[j])
                        # print(f"target:{target}")
                        # print(f"q:{q}")
                        target_tensor = torch.tensor(target, requires_grad = True).to(device)

                        loss = loss_fn(q, target_tensor)
                        # print(f"q_tensor:{q_tensor}")
                        # print(f"target_tensor:{target_tensor}")
                        optimizer.zero_grad()
                        loss.backward()

                        for param in Q.parameters():
                             param.grad.data.clamp_(-1, 1)

                        # print(f"target:{q.grad}")
                        # print(f"target:{target.grad}")
                        optimizer.step()
                        total_loss += loss.detach().cpu().numpy()

                if total_step % update_q_freq == 0:
                    Q_ast = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min :
                epsilon = epsilon - 0.00001

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1
            from statistics import mean

        if len(qvalue_memory)>0:
           qvalue_ave.append(mean(qvalue_memory))
        else:
           qvalue_ave.append(0)
        reward_ave.append(mean(reward_memory))
        profit_ave.append(env.all_money/env.initial_money)

        # print(act_memory)
        # print(reward_memory)
        total_rewards.append(total_reward)
        total_losses.append(total_loss)
        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            elapsed_time = time.time()-start
            print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
            start = time.time()

    return Q, total_losses, total_rewards, qvalue_ave, reward_ave, profit_ave