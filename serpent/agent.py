import numpy as np

from .net import MLP


class ReplayBuffer:
    """Fixed-size ring buffer backed by preallocated numpy arrays."""

    def __init__(self, capacity, state_dim, seed=0):
        self.capacity = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done):
        i = self.idx
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = s, a, r, s2, done
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        ix = self.rng.integers(0, self.size, size=n)
        return self.s[ix], self.a[ix], self.r[ix], self.s2[ix], self.done[ix]

    def __len__(self):
        return self.size


def huber_grad(err, delta=1.0):
    return np.clip(err, -delta, delta)


class DQNAgent:
    """Double DQN with a periodically synced target network."""

    def __init__(self, state_dim=11, n_actions=3, hidden=(128, 128), gamma=0.9, lr=1e-3,
                 batch_size=128, buffer_size=50_000, target_sync=400, seed=0):
        sizes = [state_dim, *hidden, n_actions]
        self.online = MLP(sizes, lr=lr, seed=seed)
        self.target = MLP(sizes, lr=lr, seed=seed)
        self.target.copy_from(self.online)
        self.memory = ReplayBuffer(buffer_size, state_dim, seed=seed)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.updates = 0
        self.rng = np.random.default_rng(seed + 1)

    def act(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.online.forward(state)[0]))

    def remember(self, *transition):
        self.memory.push(*transition)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        s, a, r, s2, done = self.memory.sample(self.batch_size)

        # double DQN: online net picks the next action, target net scores it.
        # plain DQN overestimates badly here because "keep going straight"
        # looks great right up until the wall.
        best_next = np.argmax(self.online.forward(s2), axis=1)
        next_q = self.target.forward(s2)[np.arange(len(a)), best_next]
        target = r + self.gamma * next_q * (1.0 - done)

        q = self.online.forward(s)  # must be the last forward before backward()
        rows = np.arange(len(a))
        err = q[rows, a] - target

        grad = np.zeros_like(q)
        grad[rows, a] = huber_grad(err) / len(a)
        self.online.apply(self.online.backward(grad))

        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.copy_from(self.online)
        return float(np.mean(err ** 2))
