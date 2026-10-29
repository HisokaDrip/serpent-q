import time

import numpy as np

from .agent import DQNAgent
from .game import SnakeGame


def epsilon_at(episode, total, start=1.0, end=0.01, explore_frac=0.6):
    """Linear decay over the first `explore_frac` of training, then flat."""
    horizon = max(1, int(total * explore_frac))
    return end if episode >= horizon else start + (end - start) * episode / horizon


def train(episodes=300, size=12, seed=0, learn_every=1, log_every=25, agent=None, quiet=False):
    game = SnakeGame(size, size, seed=seed)
    agent = agent or DQNAgent(seed=seed)
    scores = []
    t0 = time.time()

    for ep in range(episodes):
        eps = epsilon_at(ep, episodes)
        state = game.reset()
        done = False
        while not done:
            action = agent.act(state, eps)
            nxt, reward, done = game.step(action)
            agent.remember(state, action, reward, nxt, float(done))
            if game.steps % learn_every == 0:
                agent.learn()
            state = nxt
        scores.append(game.score)

        if not quiet and log_every and (ep + 1) % log_every == 0:
            recent = scores[-log_every:]
            print(f"ep {ep + 1:4d}  eps {eps:.2f}  avg {np.mean(recent):5.2f}  "
                  f"best {max(scores):3d}  {time.time() - t0:5.1f}s")
    return agent, scores


def evaluate(agent, games=50, size=12, seed=1234):
    """Greedy play, no exploration. Separate seed so it's not the training food layout."""
    game = SnakeGame(size, size, seed=seed, shaping=False)
    scores = []
    for _ in range(games):
        state, done = game.reset(), False
        while not done:
            state, _, done = game.step(agent.act(state, 0.0))
        scores.append(game.score)
    return np.array(scores)


def random_baseline(games=50, size=12, seed=1234):
    rng = np.random.default_rng(seed)
    game = SnakeGame(size, size, seed=seed, shaping=False)
    scores = []
    for _ in range(games):
        game.reset()
        done = False
        while not done:
            _, _, done = game.step(int(rng.integers(3)))
        scores.append(game.score)
    return np.array(scores)
