"""Task 4 (env + TD update), Task 6 (DQN loss), Task 7 (two-robot) tests. Run: python tests.py"""
import numpy as np
from lab import *

# ---- Task 4: single-robot environment table ----
for p, a, exp in [((0, 0), 0, ((0, 0), -1, False)), ((1, 1), 3, ((1, 1), -1, False)),
                  ((2, 1), 3, ((2, 2), -0.1, False)), ((2, 1), 4, ((2, 1), -0.1, False)),
                  ((2, 3), 3, ((2, 4), 10, True))]:
    got = move1(p, a); assert got == exp, (p, a, got, exp)
    print(f"env  {p} {NAMES[a]:5s} -> {got}")
env = Env1(); env.reset(); out = [env.step(4) for _ in range(40)]
assert out[-1][3] and not out[-1][2] and not any(o[3] for o in out[:-1]); print("env  40 waits -> truncated at step 40, not terminated")

# ---- Task 4: TD update by hand: Q=.4 r=1 maxQ'=2 alpha=.2 gamma=.95 ----
def td(r, term):
    Q = np.zeros((2, 5)); Q[0, 0] = .4; Q[1, 0] = 2.0
    y = q_update(Q, 0, 0, r, 1, term); return y, Q[0, 0]
for name, r, term, ey, eq in [("non-terminal", 1, False, 2.9, 0.9), ("terminal r=5", 5, True, 5.0, 1.32),
                              ("truncated (bootstrap)", 1, False, 2.9, 0.9)]:
    y, q = td(r, term); assert np.isclose(y, ey) and np.isclose(q, eq); print(f"TD   {name:22s} y={y:.3f}  Q_new={q:.3f}")

# ---- Task 6: DQN loss on hand-computed batches ----
rng = np.random.default_rng(0); net = MLP([25, 64, 64, 5], rng); tgt = net.copy()
class Const:  # target net stub: max_a' Q_target(s') = 2
    def q(self, X): return np.tile([2., 0, 0, 0, 0], (len(X), 1))
s = np.eye(25)[:3]; a = np.array([1, 3, 4]); qsa = net.q(s)[np.arange(3), a]
for name, r, d, y in [("non-terminal", 1., 0., 2.9), ("terminal", 5., 1., 5.), ("truncated", 1., 0., 2.9)]:
    L, _ = dqn_loss_and_grads(net, Const(), s, a, np.full(3, r), s, np.full(3, d))
    assert np.isclose(L, np.mean((qsa - y) ** 2)); print(f"DQN  {name:12s} y={y}  loss={L:.4f}")
# gradient only on recorded action's output + finite-difference check of backprop; target untouched
_, g = dqn_loss_and_grads(net, Const(), s, a, np.ones(3), s, np.zeros(3))
g_b3 = g[-1]; assert np.count_nonzero(g_b3) <= 3
tw = [w.copy() for w in tgt.W]
for (W, i, j) in [(0, 3, 5), (1, 10, 20), (2, 7, 2)]:
    f = lambda: dqn_loss_and_grads(net, Const(), s, a, np.ones(3), s, np.zeros(3))[0]
    net.W[W][i, j] += 1e-6; hi = f(); net.W[W][i, j] -= 2e-6; lo = f(); net.W[W][i, j] += 1e-6
    assert np.isclose((hi - lo) / 2e-6, g[2 * W][i, j], atol=1e-6), (W, i, j)
dqn_loss_and_grads(net, tgt, s, a, np.ones(3), s, np.zeros(3)); net.adam(g)
assert all(np.array_equal(w, w0) for w, w0 in zip(tgt.W, tw)); print("DQN  backprop matches finite differences; target net unchanged by update")

# ---- Task 7: two-robot collision rules ----
for pa, pb, aa, ab, exp in [((2, 1), (2, 3), 3, 2, ((2, 1), (2, 3), -10, True)),    # both into door
                            ((2, 1), (2, 2), 3, 2, ((2, 1), (2, 2), -10, True)),    # swap
                            ((2, 1), (2, 2), 3, 3, ((2, 2), (2, 3), -0.1, False)),  # follow into vacated cell
                            ((2, 1), (2, 2), 0, 4, ((1, 1), (2, 2), -0.1, False)),  # A steps aside
                            ((2, 1), (2, 2), 3, 4, ((2, 1), (2, 2), -10, True)),    # enter waiting robot
                            ((2, 3), (2, 1), 3, 2, ((2, 4), (2, 0), 14.9, True))]:  # both arrive: -.1+5+5+5
    nA, nB, r, term, info = move2(pa, pb, aa, ab); got = (nA, nB, round(r, 6), term)
    assert got == exp, (pa, pb, aa, ab, got); print(f"2rob A{pa} B{pb} {NAMES[aa]:5s}/{NAMES[ab]:5s} -> {got}")
def chk(pa, pb, aa, ab, r_exp, term_exp, why):
    nA, nB, r, term, _ = move2(pa, pb, aa, ab); assert round(r, 6) == r_exp and term == term_exp, (why, r, term)
    print(f"2rob {why}: r={round(r, 6)} term={term}")
chk((2, 4), (2, 3), 3, 4, -0.1, False, "A at goal asks 'right' (boundary): stays, no invalid penalty, no re-reward")
chk((2, 3), (2, 1), 3, 4, 4.9, False, "A arrives (k=1), B not yet")
chk((2, 4), (2, 1), 4, 2, 9.9, True, "B arrives with A already at goal (k=1, I=1)")
chk((0, 0), (2, 4), 0, 4, -1.1, False, "A invalid move (b=1)")
