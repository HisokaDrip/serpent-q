import numpy as np

from serpent.net import MLP


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    net = MLP([4, 6, 5, 3], seed=1)
    x = rng.normal(size=(5, 4))
    target = rng.normal(size=(5, 3))

    def loss():
        return 0.5 * float(((net.forward(x) - target) ** 2).sum())

    out = net.forward(x)
    grads = net.backward(out - target)

    eps = 1e-3
    for p, g in zip(net.params, grads):
        for idx in [tuple(rng.integers(0, s) for s in p.shape) for _ in range(4)]:
            old = p[idx]
            p[idx] = old + eps
            up = loss()
            p[idx] = old - eps
            down = loss()
            p[idx] = old
            numeric = (up - down) / (2 * eps)
            assert abs(numeric - g[idx]) < 2e-2 * max(1.0, abs(numeric))


def test_adam_fits_a_tiny_regression():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(256, 2)).astype(np.float32)
    y = (np.sin(3 * x[:, :1]) + x[:, 1:] ** 2).astype(np.float32)
    net = MLP([2, 32, 32, 1], lr=1e-2, seed=0)
    first = None
    for _ in range(400):
        pred = net.forward(x)
        mse = float(((pred - y) ** 2).mean())
        first = first if first is not None else mse
        net.apply(net.backward(2 * (pred - y) / len(x)))
    assert mse < first * 0.1


def test_save_load_roundtrip(tmp_path):
    net = MLP([3, 8, 2], seed=4)
    path = tmp_path / "w.npz"
    net.save(path)
    loaded = MLP.load(path)
    x = np.ones((2, 3), dtype=np.float32)
    np.testing.assert_allclose(net.forward(x), loaded.forward(x))
