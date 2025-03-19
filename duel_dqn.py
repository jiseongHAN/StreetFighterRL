import random
from collections import deque

import retro
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from wrappers import *


def arrange(s):
    if not type(s) == "numpy.ndarray":
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)


class replay_memory(object):
    def __init__(self, N):
        self.memory = deque(maxlen=N)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)


class model(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(model, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(20736, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)

        self.device = device
        self.seq = nn.Sequential(self.layer1, self.layer2, self.fc, self.q, self.v)

        self.seq.apply(init_weights)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - 1 / adv.shape[-1] * adv.sum(-1, keepdim=True))

        return q


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train(q, q_target, memory, batch_size, gamma, optimizer, device):
    s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))
    s = np.array(s).squeeze()
    s_prime = np.array(s_prime).squeeze()
    a_max = q(s_prime).max(1)[1].unsqueeze(-1)
    r = torch.FloatTensor(r).unsqueeze(-1).to(device)
    done = torch.FloatTensor(done).unsqueeze(-1).to(device)
    with torch.no_grad():
        y = r + gamma * q_target(s_prime).gather(1, a_max) * done
    a = torch.tensor(a).unsqueeze(-1).to(device)
    q_value = torch.gather(q(s), dim=1, index=a.view(-1, 1).long())

    loss = F.smooth_l1_loss(q_value, y).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def copy_weights(q, q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)


def main(env, q, q_target, optimizer, device):
    t = 0
    gamma = 0.99
    batch_size = 256

    N = 50000
    eps = 0.001
    memory = replay_memory(N)
    update_interval = 50
    print_interval = 10

    total_score = 0.0
    loss = 0.0

    print("Start training")
    for k in range(1000000):
        s = arrange(env.reset())
        done = False

        while not done:
            if eps > np.random.rand():
                a = env.action_space.sample()
            else:
                if device == "cpu":
                    a = np.argmax(q(s).detach().numpy())
                else:
                    a = np.argmax(q(s).cpu().detach().numpy())
            s_prime, r, done, _ = env.step(a)
            s_prime = arrange(s_prime)
            total_score += r
            r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
            memory.push((s, float(r), int(a), s_prime, int(1 - done)))
            s = s_prime
            stage = env.unwrapped.data.lookup_all()["matches_won"]
            if len(memory) > 2000:
                loss += train(q, q_target, memory, batch_size, gamma, optimizer, device)
                t += 1
            if t % update_interval == 0:
                copy_weights(q, q_target)
                torch.save(q.state_dict(), "ckpt/street.pth")
                torch.save(q_target.state_dict(), "ckpt/street_q_target.pth")

        if k % print_interval == 0:
            print(
                "%s |Epoch : %d | score : %f | loss : %.2f | match : %d"
                % (
                    device,
                    k,
                    total_score / print_interval,
                    loss / print_interval,
                    stage,
                )
            )
            total_score = 0
            loss = 0.0
        if t % update_interval == 0 and stage > 0:
            torch.save(q.state_dict(), "ckpt/street1.pth")
            torch.save(q_target.state_dict(), "ckpt/street_q_target1.pth")


if __name__ == "__main__":
    n_frame = 4
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    env = wrap_streetfighter(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = model(n_frame, env.action_space.n, device).to(device)
    q_target = model(n_frame, env.action_space.n, device).to(device)
    # q.load_state_dict(torch.load("street.pth"))
    # q_target.load_state_dict(torch.load("street_q_target.pth"))

    optimizer = optim.Adam(q.parameters(), lr=0.0001)
    print(device)
    main(env, q, q_target, optimizer, device)
