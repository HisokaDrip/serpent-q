import argparse
import os
import sys
import time

import numpy as np

from .agent import DQNAgent
from .game import SnakeGame
from .net import MLP
from .train import evaluate, random_baseline, train

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "snake.npz")


def load_agent(path):
    agent = DQNAgent()
    agent.online = MLP.load(path)
    return agent


def cmd_train(args):
    agent, scores = train(args.episodes, size=args.size, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    agent.online.save(args.out)
    print(f"\nsaved weights -> {args.out}")
    _report(agent, args.size)


def _report(agent, size):
    greedy = evaluate(agent, games=50, size=size)
    rand = random_baseline(games=50, size=size)
    print(f"greedy agent : mean {greedy.mean():5.2f}  median {np.median(greedy):4.0f}  best {greedy.max()}")
    print(f"random policy: mean {rand.mean():5.2f}  median {np.median(rand):4.0f}  best {rand.max()}")


def cmd_eval(args):
    _report(load_agent(args.weights), args.size)


def cmd_play(args):
    agent = load_agent(args.weights)
    game = SnakeGame(args.size, args.size, seed=args.seed, shaping=False)
    state, done = game.reset(), False
    while not done:
        state, _, done = game.step(agent.act(state))
        sys.stdout.write("\x1b[H\x1b[2J")  # clear screen
        print(game.render())
        print(f"score {game.score}   steps {game.steps}")
        time.sleep(args.delay)
    print("game over")


def main(argv=None):
    p = argparse.ArgumentParser(prog="serpent", description="DQN snake in plain numpy")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a new agent")
    t.add_argument("--episodes", type=int, default=300)
    t.add_argument("--size", type=int, default=12)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="snake.npz")
    t.set_defaults(func=cmd_train)

    for name, fn, helptext in [("eval", cmd_eval, "score an agent over 50 greedy games"),
                               ("play", cmd_play, "watch an agent play in the terminal")]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--weights", default=DEFAULT_WEIGHTS)
        s.add_argument("--size", type=int, default=12)
        s.add_argument("--seed", type=int, default=7)
        s.add_argument("--delay", type=float, default=0.06)
        s.set_defaults(func=fn)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
