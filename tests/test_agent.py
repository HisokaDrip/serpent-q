import os

import numpy as np

from serpent.__main__ import DEFAULT_WEIGHTS, load_agent
from serpent.agent import DQNAgent, ReplayBuffer
from serpent.train import epsilon_at, evaluate, random_baseline, train


def test_replay_buffer_wraps():
    buf = ReplayBuffer(5, 2)
    for i in range(8):
        buf.push([i, i], i % 3, float(i), [i + 1, i + 1], 0.0)
    assert len(buf) == 5
    assert set(buf.r.tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0}


def test_epsilon_schedule():
    assert epsilon_at(0, 100) == 1.0
    assert epsilon_at(60, 100) == 0.01
    assert 0.01 < epsilon_at(30, 100) < 1.0


def test_target_network_syncs():
    agent = DQNAgent(batch_size=8, target_sync=3, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        agent.remember(rng.random(11), int(rng.integers(3)), 1.0, rng.random(11), 0.0)
    for _ in range(3):
        agent.learn()
    for a, b in zip(agent.online.params, agent.target.params):
        np.testing.assert_array_equal(a, b)


def test_short_training_beats_random():
    # ~150 episodes is enough to clearly learn "go to food, avoid walls"
    agent, scores = train(150, seed=0, quiet=True)
    trained = evaluate(agent, games=20)
    rand = random_baseline(games=20)
    assert trained.mean() > rand.mean() + 2


def test_shipped_weights_play_well():
    assert os.path.exists(DEFAULT_WEIGHTS)
    scores = evaluate(load_agent(DEFAULT_WEIGHTS), games=20)
    assert scores.mean() > 10
