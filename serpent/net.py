import numpy as np


class MLP:
    """Small fully-connected net with ReLU hidden layers and a linear head.

    Backprop and Adam are done by hand. It's ~80 lines and that's the point:
    nothing here is hidden behind a framework.
    """

    def __init__(self, sizes, lr=1e-3, seed=0, betas=(0.9, 0.999), eps=1e-8):
        rng = np.random.default_rng(seed)
        # He init since everything except the head is ReLU
        self.W = [rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)).astype(np.float32)
                  for fan_in, fan_out in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n, dtype=np.float32) for n in sizes[1:]]
        self.sizes = list(sizes)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.adam_eps = eps
        self.t = 0
        self._m = [np.zeros_like(p) for p in self.params]
        self._v = [np.zeros_like(p) for p in self.params]
        self._acts = None

    @property
    def params(self):
        return self.W + self.b

    def forward(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        acts = [x]
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < last:
                x = np.maximum(x, 0.0)
            acts.append(x)
        self._acts = acts
        return x

    def backward(self, grad_out):
        """Gradients w.r.t. every param, given dL/d(output) of the last forward()."""
        n = len(self.W)
        gW, gb = [None] * n, [None] * n
        g = grad_out.astype(np.float32)
        for i in reversed(range(n)):
            gW[i] = self._acts[i].T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (self._acts[i] > 0)
        return gW + gb

    def apply(self, grads, clip=10.0):
        # global-norm clipping; Q-targets can spike early on and blow things up
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if clip and norm > clip:
            grads = [g * (clip / norm) for g in grads]

        self.t += 1
        b1, b2 = self.beta1, self.beta2
        lr_t = self.lr * np.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)
        for p, g, m, v in zip(self.params, grads, self._m, self._v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            p -= lr_t * m / (np.sqrt(v) + self.adam_eps)
        return norm

    def copy_from(self, other):
        for dst, src in zip(self.params, other.params):
            dst[...] = src

    def save(self, path):
        arrays = {f"W{i}": W for i, W in enumerate(self.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez(path, sizes=np.array(self.sizes), **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        net = cls([int(s) for s in data["sizes"]])
        for i in range(len(net.W)):
            net.W[i][...] = data[f"W{i}"]
            net.b[i][...] = data[f"b{i}"]
        return net
