# serpent-q

A Deep Q-Network that teaches itself Snake, written in plain NumPy. No PyTorch, no gym, no TensorFlow. The game, the neural net (forward pass, backprop, Adam) and the RL loop are all in this repo and add up to roughly 400 lines.

It goes from moving at random to scoring 20-40 on a 12x12 board in about 300 games, which takes under 20 seconds on a laptop CPU.

```
+------------------------+
|· · · · · · · · · · · · |
|· · · · · · · · · · · · |
|· · · o · · · · · · · · |
|· · · o · · · · · · · · |
|· · o o · · · · · · · · |
|· · o · · · · · · · · · |
|· · o · · · · · · · · · |
|o o o · · · · · · · · · |
|o o o o o o o o o · · ● |
|@ o o o o o o o o · · · |
+------------------------+
score 25   steps 242
```

## How it works

**State.** The agent doesn't see pixels. Each step it gets 11 yes/no features: is there danger straight ahead, to the right, to the left; which way it's heading; and whether the food is left, right, above or below. That's a tiny input, but it's enough to learn a good greedy strategy fast.

**Actions.** These are relative to the snake: go straight, turn right, or turn left. With relative actions the snake can never try to reverse into itself, which removes a whole class of stupid deaths.

**Rewards.** It gets +10 for eating and -10 for dying. There's also a small shaping term (+0.1 for moving closer to the food, -0.15 for moving away), which mostly helps in the first hundred games. A starvation limit kills it if it loops forever without eating.

**Learning.** This is Double DQN with experience replay and a target network that syncs every 400 updates, a 11 → 128 → 128 → 3 MLP, Huber loss and gradient-norm clipping. Plain DQN overestimated Q-values a lot here ("going straight is great" right up until the wall), and switching to Double DQN made training much more stable.

```
serpent/
  game.py    snake environment + feature extraction
  net.py     MLP with hand-written backprop and Adam
  agent.py   replay buffer + double DQN
  train.py   training loop, greedy eval, random baseline
```

## Running it

```bash
pip install -r requirements.txt

# watch the pretrained agent play (weights/snake.npz)
python -m serpent play

# train your own from scratch
python -m serpent train --episodes 300 --out my_snake.npz
python -m serpent play --weights my_snake.npz

# score it over 50 games
python -m serpent eval
```

A real training run (seed 0):

```
ep   25  eps 0.87  avg  0.32  best   2    0.4s
ep   50  eps 0.73  avg  0.56  best   2    0.7s
ep  100  eps 0.46  avg  1.48  best   3    1.7s
ep  150  eps 0.18  avg  4.20  best  11    3.2s
ep  175  eps 0.04  avg  5.88  best  16    4.2s
ep  200  eps 0.01  avg 17.48  best  33    7.3s
ep  250  eps 0.01  avg 20.36  best  33   12.6s
ep  300  eps 0.01  avg 18.80  best  44   18.4s

saved weights -> weights/snake.npz
greedy agent : mean 24.60  median   24  best 47
random policy: mean  0.18  median    0  best 2
```

Around episode 175 exploration bottoms out, and it suddenly works out that it can just keep eating.

## Tests

```bash
pip install pytest
pytest -q
```

The tests cover the game rules (walls, growing, the tail-chasing edge case), a finite-difference gradient check on the MLP, the replay buffer and target sync, and a short training run that has to clearly beat the random baseline.

## What I'd do next

The agent's main weakness is that it can't see its own body beyond one cell, so long snakes eventually trap themselves. Next I'd try:
- adding flood-fill or "reachable area" features so it can tell when a move leads into a dead end
- switching to a small CNN over the whole grid (that would mean writing conv backprop in numpy too)
- prioritized replay, since most transitions are boring and deaths are rare
