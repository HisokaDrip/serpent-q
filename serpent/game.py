from collections import deque

import numpy as np

# clockwise, so "turn right" is +1 and "turn left" is -1 (mod 4)
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

STRAIGHT, RIGHT, LEFT = 0, 1, 2


class SnakeGame:
    """Grid snake with a relative action space (straight / right / left).

    Relative actions mean the agent can never pick "reverse into yourself",
    which cuts out a whole class of dumb deaths and makes learning a lot faster.
    """

    def __init__(self, width=12, height=12, seed=None, shaping=True):
        self.w, self.h = width, height
        self.rng = np.random.default_rng(seed)
        self.shaping = shaping
        self.reset()

    def reset(self):
        cx, cy = self.w // 2, self.h // 2
        self.direction = 1
        self.body = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self._place_food()
        return self.state()

    @property
    def head(self):
        return self.body[0]

    def _place_food(self):
        taken = set(self.body)
        free = [(x, y) for x in range(self.w) for y in range(self.h) if (x, y) not in taken]
        self.food = free[self.rng.integers(len(free))] if free else None

    def collides(self, pt):
        x, y = pt
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return True
        # the tail slides forward on the same tick, so stepping onto it is fine
        return pt in list(self.body)[:-1]

    def _dist_to_food(self):
        if self.food is None:
            return 0
        return abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

    def step(self, action):
        if self.done:
            raise RuntimeError("game is over, call reset()")

        if action == RIGHT:
            self.direction = (self.direction + 1) % 4
        elif action == LEFT:
            self.direction = (self.direction - 1) % 4

        dx, dy = DIRS[self.direction]
        new_head = (self.head[0] + dx, self.head[1] + dy)
        before = self._dist_to_food()

        self.steps += 1
        self.steps_since_food += 1

        # starvation limit stops the agent from learning to circle forever
        starving = self.steps_since_food > 100 * len(self.body)
        if self.collides(new_head) or starving:
            self.done = True
            return self.state(), -10.0, True

        self.body.appendleft(new_head)
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            self._place_food()
            if self.food is None:  # filled the board, absurdly rare but possible
                self.done = True
            return self.state(), 10.0, self.done

        self.body.pop()
        reward = 0.0
        if self.shaping:
            # tiny nudge towards the food; big enough to speed up early learning,
            # small enough that it never outweighs dying
            reward = 0.1 if self._dist_to_food() < before else -0.15
        return self.state(), reward, False

    def state(self):
        hx, hy = self.head
        d = self.direction

        def ahead(direction):
            dx, dy = DIRS[direction % 4]
            return (hx + dx, hy + dy)

        danger = [self.collides(ahead(d)), self.collides(ahead(d + 1)), self.collides(ahead(d - 1))]
        heading = [d == i for i in range(4)]
        fx, fy = self.food if self.food else (hx, hy)
        food = [fx < hx, fx > hx, fy < hy, fy > hy]
        return np.array(danger + heading + food, dtype=np.float32)

    def render(self):
        grid = [["·"] * self.w for _ in range(self.h)]
        if self.food:
            grid[self.food[1]][self.food[0]] = "●"
        for i, (x, y) in enumerate(self.body):
            if 0 <= x < self.w and 0 <= y < self.h:
                grid[y][x] = "@" if i == 0 else "o"
        border = "+" + "-" * (self.w * 2) + "+"
        rows = ["|" + " ".join(r) + " |" for r in grid]
        return "\n".join([border, *rows, border])
