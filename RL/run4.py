import pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lab import *
def load(f):
    W, b = pickle.load(open(f, "rb")); n = object.__new__(MLP); n.W, n.b = W, b; return n
env = Env2()
def table(ev):
    print(" t | A     | B     | joint action"); 
    for t, (pa, pb) in enumerate(ev["path"]): print(f"{t:2d} | {pa} | {pb}")
def plot(ev, fn, title):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for r, c in OBST: ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="gray"))
    for k, col, nm, off in ((0, "C0", "A", -.18), (1, "C3", "B", .18)):
        P = [p[k] for p in ev["path"]]; ax.plot([c + off for r, c in P], [r + off for r, c in P], "o-", c=col, ms=11, alpha=.5, label=f"robot {nm}")
        for t, (r, c) in enumerate(P): ax.text(c + off, r + off + .05, str(t), ha="center", fontsize=7, color="k")
    ax.set(xlim=(-.5, 4.5), ylim=(4.5, -.5), xticks=range(5), yticks=range(5), title=title); ax.grid(True); ax.legend(loc="lower right", fontsize=8)
    plt.savefig(fn, dpi=110, bbox_inches="tight")
print("=== 200k seed 8 (successful) ==="); n8 = load("net2_200000_8.pkl"); e8 = eval_net(env, n8); table(e8); plot(e8, "two_traj_200k_s8.png", "Greedy joint trajectory (200k steps, seed 8), labels = time step")
print("joint actions:", [divmod(int(np.argmax(n8.q(env.enc(s)[None])[0])), 5) for s in e8["path"][:-1]])
print("=== 60k seed 7 (spec run, failed) ==="); n7 = load("net2_seed7.pkl"); e7 = eval_net(env, n7); plot(e7, "two_traj_60k_s7_fail.png", "FAILED run: 60k steps, seed 7")
print(e7["path"][:6], "... oscillates", e7["path"][-2:])
print("=== alt start, 200k seed 10 ==="); n10 = load("net2_200000_10.pkl"); ex = eval_net(env, n10, ((0, 0), (4, 4))); table(ex)
