import numpy as np
import pytest

from serpent.game import LEFT, RIGHT, STRAIGHT, SnakeGame


def test_initial_state_shape_and_heading():
    g = SnakeGame(10, 10, seed=0)
    s = g.state()
    assert s.shape == (11,)
    assert s[3:7].tolist() == [0, 1, 0, 0]  # facing right
    assert len(g.body) == 3


def test_eating_grows_and_scores():
    g = SnakeGame(10, 10, seed=0)
    hx, hy = g.head
    g.food = (hx + 1, hy)
    _, reward, done = g.step(STRAIGHT)
    assert reward == 10.0 and not done
    assert g.score == 1 and len(g.body) == 4
    assert g.food not in g.body


def test_wall_kills():
    g = SnakeGame(6, 6, seed=0)
    g.food = (0, 0)
    done = False
    for _ in range(10):
        _, reward, done = g.step(STRAIGHT)
        if done:
            break
    assert done and reward == -10.0


def test_moving_into_old_tail_cell_is_legal():
    g = SnakeGame(10, 10, seed=0)
    # 2x2 loop: head follows the tail around, which should never count as a hit
    g.body.clear()
    g.body.extend([(5, 5), (4, 5), (4, 6), (5, 6)])
    g.direction = 1
    g.food = (0, 0)
    _, _, done = g.step(RIGHT)  # turn down into (5, 6), which is the tail
    assert not done


def test_turn_directions():
    g = SnakeGame(10, 10, seed=0)
    g.food = (0, 0)
    g.step(RIGHT)
    assert g.direction == 2  # was right, now down
    g.step(LEFT)
    assert g.direction == 1


def test_step_after_game_over_raises():
    g = SnakeGame(4, 4, seed=0)
    while not g.done:
        g.step(STRAIGHT)
    with pytest.raises(RuntimeError):
        g.step(STRAIGHT)


def test_danger_flags():
    g = SnakeGame(10, 10, seed=0)
    g.body.clear()
    g.body.extend([(9, 0), (8, 0), (7, 0)])
    g.direction = 1
    danger = g.state()[:3]
    # wall ahead and to the left (up), free to the right (down)
    assert danger.tolist() == [1, 0, 1]
