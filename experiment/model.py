"""Numpy implementation of a Double-DQN agent with parameter sharing across all
civilizations (self-play). Deliberately dependency-free (no torch): the network is
a 24->64->64->5 MLP, small enough that hand-written forward/backward on CPU is
faster than framework overhead on the 2-vCPU sandbox.

Components:
  MLP           — 2-hidden-layer ReLU network, He init, manual backprop
  Adam          — standard Adam optimizer over a dict of parameters
  ReplayBuffer  — preallocated ring buffer (state, action, reward, next_state, done)
  DQNAgent      — online/target nets, epsilon-greedy with action masking,
                  Double-DQN targets, Huber loss, global-norm grad clipping
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.sizes = sizes
        self.params: dict[str, np.ndarray] = {}
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            w = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(sizes[i], sizes[i + 1]))
            self.params[f"W{i}"] = w.astype(np.float64)
            self.params[f"b{i}"] = np.zeros(sizes[i + 1], dtype=np.float64)
        self.n_layers = len(sizes) - 1

    def forward(self, x: np.ndarray):
        """x: (B, D) -> returns output (B, A) and cache for backward."""
        cache = {"a0": x}
        h = x
        for i in range(self.n_layers):
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            if i < self.n_layers - 1:
                h = np.maximum(z, 0.0)
                cache[f"z{i+1}"] = z
                cache[f"a{i+1}"] = h
            else:
                cache["out"] = z
        return cache["out"], cache

    def backward(self, cache: dict, grad_out: np.ndarray) -> dict[str, np.ndarray]:
        """grad_out: (B, A) dL/doutput. Returns gradients dict."""
        grads: dict[str, np.ndarray] = {}
        g = grad_out
        for i in range(self.n_layers - 1, -1, -1):
            a_prev = cache[f"a{i}"]
            grads[f"W{i}"] = a_prev.T @ g
            grads[f"b{i}"] = g.sum(axis=0)
            if i > 0:
                g = g @ self.params[f"W{i}"].T
                g = g * (cache[f"z{i}"] > 0)
        return grads


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 6e-4,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]):
        self.t += 1
        b1t = 1.0 - self.b1 ** self.t
        b2t = 1.0 - self.b2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            params[k] -= self.lr * (self.m[k] / b1t) / (np.sqrt(self.v[k] / b2t) + self.eps)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0,
                 n_valid: int = 10):     # v2.4: valid-mask width parametrised (was hardcoded 10)
        self.capacity = capacity
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.v2 = np.ones((capacity, n_valid), dtype=np.float32)   # next-state valid mask
        self.idx = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, d, v2=None):
        """v2.2 audit fix: v2 = next-state valid-action mask (B,8); defaults to all-valid."""
        n = len(a)
        if v2 is None:
            v2 = np.ones((n, self.v2.shape[1]), dtype=np.float32)
        end = self.idx + n
        if end <= self.capacity:
            self.s[self.idx:end] = s
            self.a[self.idx:end] = a
            self.r[self.idx:end] = r
            self.s2[self.idx:end] = s2
            self.d[self.idx:end] = d
            self.v2[self.idx:end] = v2
        else:  # wrap
            k = self.capacity - self.idx
            self.s[self.idx:] = s[:k]
            self.a[self.idx:] = a[:k]
            self.r[self.idx:] = r[:k]
            self.s2[self.idx:] = s2[:k]
            self.d[self.idx:] = d[:k]
            self.v2[self.idx:] = v2[:k]
            self.s[: n - k] = s[k:]
            self.a[: n - k] = a[k:]
            self.r[: n - k] = r[k:]
            self.s2[: n - k] = s2[k:]
            self.d[: n - k] = d[k:]
            self.v2[: n - k] = v2[k:]
        self.idx = end % self.capacity
        if end >= self.capacity:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch: int):
        idx = self.rng.integers(0, len(self), size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx], self.v2[idx]


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int],
                 lr: float, gamma: float, batch_size: int, replay_size: int,
                 target_sync: int, grad_clip: float = 5.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.grad_clip = grad_clip
        self.online = MLP([obs_dim] + hidden + [n_actions], rng)
        self.target = MLP([obs_dim] + hidden + [n_actions], rng)
        self.target.params = {k: v.copy() for k, v in self.online.params.items()}
        self.opt = Adam(self.online.params, lr=lr)
        self.buffer = ReplayBuffer(replay_size, obs_dim, seed=seed + 1, n_valid=n_actions)
        self.rng = np.random.default_rng(seed + 2)
        self.updates = 0

    # ------------------------------------------------------------------ act
    def act(self, obs: np.ndarray, valid: np.ndarray, epsilon: float) -> np.ndarray:
        """Epsilon-greedy over valid actions. obs (B,D), valid (B,A) bool."""
        q, _ = self.online.forward(obs.astype(np.float64))
        q = np.where(valid, q, -np.inf)
        greedy = np.argmax(q, axis=1)
        explore = self.rng.uniform(size=len(obs)) < epsilon
        if explore.any():
            for i in np.where(explore)[0]:
                opts = np.where(valid[i])[0]
                greedy[i] = self.rng.choice(opts)
        return greedy

    # --------------------------------------------------------------- update
    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, d, v2 = self.buffer.sample(self.batch_size)
        s, s2 = s.astype(np.float64), s2.astype(np.float64)
        b = len(s)

        # Double-DQN target: action chosen by online, evaluated by target.
        # v2.2 audit fix: mask the argmax by the NEXT state's valid-action mask so
        # the target policy agrees with the (masked) behavior policy.
        q_next_online, _ = self.online.forward(s2)
        q_next_online = np.where(d[:, None] > 0, -np.inf, q_next_online)
        q_next_online = np.where(v2 > 0, q_next_online, -np.inf)
        a_star = np.argmax(q_next_online, axis=1)
        q_next_target, _ = self.target.forward(s2)
        q_star = q_next_target[np.arange(b), a_star]
        y = r + self.gamma * (1.0 - d) * q_star

        q, cache = self.online.forward(s)
        q_sa = q[np.arange(b), a]
        err = q_sa - y
        loss = float(np.mean(np.where(np.abs(err) <= 1, 0.5 * err**2, np.abs(err) - 0.5)))

        grad_q = np.zeros_like(q)
        grad_q[np.arange(b), a] = np.clip(err, -1.0, 1.0) / b
        grads = self.online.backward(cache, grad_q)

        # global-norm clipping
        total = np.sqrt(sum(float((g**2).sum()) for g in grads.values()))
        if total > self.grad_clip:
            scale = self.grad_clip / (total + 1e-12)
            for k in grads:
                grads[k] *= scale
        self.opt.step(self.online.params, grads)

        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.sync_target()
        return loss

    def sync_target(self):
        self.target.params = {k: v.copy() for k, v in self.online.params.items()}

    # ----------------------------------------------------------------- io
    def save(self, path: str):
        np.savez(path, **{f"online__{k}": v for k, v in self.online.params.items()},
                 updates=self.updates)

    def load(self, path: str):
        data = np.load(path)
        for k in list(self.online.params):
            self.online.params[k] = data[f"online__{k}"]
        self.updates = int(data["updates"])
        self.sync_target()
