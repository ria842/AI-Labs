"""RL lab: tabular Q-learning, DQN (NumPy), one- and two-robot warehouse. Seed 7 throughout."""
import numpy as np

# ---------------- map & single-robot environment ----------------
OBST = {(0, 2), (1, 2), (3, 2), (4, 2)}
S, G = (2, 0), (2, 4)
GAMMA, T_MAX = 0.95, 40
DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up, down, left, right, wait
NAMES = ["up", "down", "left", "right", "wait"]
idx = lambda p: 5 * p[0] + p[1]


def slide(p, a):
    """Apply action a at cell p. Returns (new_cell, valid). Obstacle/boundary -> stay, invalid."""
    q = (p[0] + DELTA[a][0], p[1] + DELTA[a][1])
    if not (0 <= q[0] < 5 and 0 <= q[1] < 5) or q in OBST:
        return p, False
    return q, True


def move1(p, a):
    """Test-mode single transition: (pos, action) -> (next_pos, reward, terminated)."""
    q, ok = slide(p, a)
    if q == G:
        return q, 10.0, True
    return q, (-0.1 if ok else -1.0), False


class Env1:
    nA, obs_dim = 5, 25

    def __init__(self, start=S):
        self.start = start

    def reset(self, pos=None):
        self.s, self.t = pos or self.start, 0
        return self.s

    def step(self, a):
        s2, r, term = move1(self.s, a)
        self.t += 1
        info = {"invalid": r == -1.0}
        self.s = s2
        return s2, r, term, (self.t >= T_MAX and not term), info

    @staticmethod
    def enc(s):
        v = np.zeros(25); v[idx(s)] = 1; return v


# ---------------- tabular Q-learning ----------------
def eps_greedy(Q, s, eps, rng):
    """Random action w.p. eps, else argmax with RANDOM tie-break (training)."""
    if rng.random() < eps:
        return int(rng.integers(5))
    q = Q[idx(s)]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def greedy(Q, s):
    """Evaluation: deterministic argmax, ties -> lowest action index."""
    return int(np.argmax(Q[idx(s)]))


def q_update(Q, i, a, r, i2, term, alpha=0.2, gamma=GAMMA):
    """One TD update. Target drops bootstrap ONLY for true termination (not truncation)."""
    y = r if term else r + gamma * Q[i2].max()
    Q[i, a] += alpha * (y - Q[i, a])
    return y


def q_train(episodes=1500, seed=7, eps0=1.0, eps1=0.05, decay=1200, alpha=0.2):
    rng, Q, env = np.random.default_rng(seed), np.zeros((25, 5)), Env1()
    log = {k: [] for k in ("ret", "success", "eps", "invalid", "timeout")}
    for ep in range(episodes):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / decay)
        s, R, inv, term, trunc = env.reset(), 0.0, 0, False, False
        while not (term or trunc):
            a = eps_greedy(Q, s, eps, rng)
            s2, r, term, trunc, info = env.step(a)
            q_update(Q, idx(s), a, r, idx(s2), term, alpha)  # s2 is pre-reset next state
            R, inv, s = R + r, inv + info["invalid"], s2
        for k, v in zip(log, (R, term, eps, inv, trunc)):
            log[k].append(v)
    return Q, {k: np.array(v) for k, v in log.items()}


def rollout(step_fn, s, policy, T=T_MAX):
    """Generic greedy/any-policy rollout with env-like object. Returns dict."""
    path, inv, term, trunc, coll = [s], 0, False, False, False
    for _ in range(T):
        s, r, term, trunc, info = step_fn(policy(s))
        path.append(s); inv += info["invalid"]; coll = info.get("collision", False)
        if term or trunc:
            break
    return dict(path=path, steps=len(path) - 1, success=term and not coll, collision=coll,
                timeout=trunc, invalid=inv)


def eval1(Q, start=S):
    env = Env1(start); s = env.reset()
    return rollout(env.step, s, lambda s: greedy(Q, s))


def random_baseline(n=100, seed=7):
    rng, out = np.random.default_rng(seed), []
    for _ in range(n):
        env = Env1(); s = env.reset()
        out.append(rollout(env.step, s, lambda s: int(rng.integers(5))))
    return out


# ---------------- NumPy MLP + DQN ----------------
class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2 / i), (i, o)) for i, o in zip(sizes, sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.params()]; self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self):
        return [p for wb in zip(self.W, self.b) for p in wb]

    def copy(self):
        n = object.__new__(MLP); n.W = [w.copy() for w in self.W]; n.b = [b.copy() for b in self.b]; return n

    def forward(self, X):
        hs = [X]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            hs.append(np.maximum(z, 0) if k < len(self.W) - 1 else z)  # last layer linear
        return hs

    def q(self, X):
        return self.forward(X)[-1]

    def grads(self, hs, dout):
        gW, gb = [], []
        for k in reversed(range(len(self.W))):
            gW.insert(0, hs[k].T @ dout); gb.insert(0, dout.sum(0))
            if k:
                dout = (dout @ self.W[k].T) * (hs[k] > 0)
        return [g for p in zip(gW, gb) for g in p]

    def adam(self, grads, lr=1e-3, b1=0.9, b2=0.999, e=1e-8):
        self.t += 1
        for p, g, m, v in zip(self.params(), grads, self.m, self.v):
            m *= b1; m += (1 - b1) * g; v *= b2; v += (1 - b2) * g * g
            p -= lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + e)  # in-place


def td_targets(r, done, qmax_next, gamma=GAMMA):
    """y = r + gamma*(1-done)*max Q_target(s'). done=1 ONLY for true termination."""
    return r + gamma * (1 - done) * qmax_next


def dqn_loss_and_grads(net, tgt, s, a, r, s2, done, gamma=GAMMA):
    """MSE between Q_theta(s,a_recorded) and detached target from the TARGET network."""
    y = td_targets(r, done, tgt.q(s2).max(1), gamma)          # no gradient path: plain arrays from tgt
    hs = net.forward(s); q = hs[-1]
    rows = np.arange(len(a)); qsa = q[rows, a]                 # <- recorded action selected here
    d = np.zeros_like(q); d[rows, a] = 2 * (qsa - y) / len(a)  # dL/dq nonzero only at recorded action
    return np.mean((qsa - y) ** 2), net.grads(hs, d)


def dqn_train(env, steps, seed=7, hidden=64, cap=20000, batch=64, warm=500, every=4,
              sync=250, lr=1e-3, frac=0.8, log_every=None):
    rng = np.random.default_rng(seed)
    net = MLP([env.obs_dim, hidden, hidden, env.nA], rng); tgt = net.copy()
    B = dict(s=np.zeros((cap, env.obs_dim)), a=np.zeros(cap, int), r=np.zeros(cap),
             s2=np.zeros((cap, env.obs_dim)), d=np.zeros(cap))
    n = upd = 0
    s, R, hist = env.reset(), 0.0, dict(ep_ret=[], ep_end=[], loss=[])
    for t in range(steps):
        eps = 1 - 0.95 * min(1, t / (frac * steps))
        e = env.enc(s)
        a = int(rng.integers(env.nA)) if rng.random() < eps else int(np.argmax(net.q(e[None])[0]))
        s2, r, term, trunc, _ = env.step(a)
        i = n % cap; B["s"][i], B["a"][i], B["r"][i], B["s2"][i], B["d"][i] = e, a, r, env.enc(s2), term
        n += 1; R += r
        s = env.reset() if (term or trunc) else s2
        if term or trunc:
            hist["ep_ret"].append(R); hist["ep_end"].append(t); R = 0.0
        if n >= warm and (t + 1) % every == 0:
            j = rng.integers(0, min(n, cap), batch)
            loss, g = dqn_loss_and_grads(net, tgt, B["s"][j], B["a"][j], B["r"][j], B["s2"][j], B["d"][j])
            net.adam(g, lr); upd += 1; hist["loss"].append(loss)
            if upd % sync == 0:
                tgt = net.copy()
    hist["updates"] = upd
    return net, hist


def greedy_net(net, env):
    return lambda s: int(np.argmax(net.q(env.enc(s)[None])[0]))  # ties -> lowest index


# ---------------- two-robot environment ----------------
SA, GA, SB, GB = (2, 0), (2, 4), (2, 4), (2, 0)


def move2(pA, pB, aA, aB):
    """Simultaneous step. Returns (newA, newB, reward, terminated, info)."""
    nA, vA = (pA, True) if pA == GA else slide(pA, aA)   # robot at goal stays, no invalid penalty
    nB, vB = (pB, True) if pB == GB else slide(pB, aB)
    if nA == nB or (nA == pB and nB == pA):              # same-cell or swap collision
        return pA, pB, -10.0, True, dict(collision=True, invalid=0)
    b = (not vA) + (not vB)
    k = (nA == GA and pA != GA) + (nB == GB and pB != GB)
    I = int(nA == GA and nB == GB)
    return nA, nB, -0.1 - b + 5 * k + 5 * I, bool(I), dict(collision=False, invalid=b)


class Env2:
    nA, obs_dim = 25, 50

    def __init__(self, start=(SA, SB)):
        self.start = start

    def reset(self, s=None):
        self.s, self.t = s or self.start, 0
        return self.s

    def step(self, j):
        aA, aB = divmod(j, 5)
        nA, nB, r, term, info = move2(*self.s, aA, aB)
        self.s = (nA, nB); self.t += 1
        return self.s, r, term, (self.t >= T_MAX and not term), info

    @staticmethod
    def enc(s):
        v = np.zeros(50); v[idx(s[0])] = 1; v[25 + idx(s[1])] = 1; return v


def eval_net(env, net, start=None):
    """Greedy rollout (eps=0, no updates) from `start` (default: env.start)."""
    return rollout(env.step, env.reset(start), greedy_net(net, env))
