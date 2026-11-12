import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lab import *
W = 50
ma = lambda x: np.convolve(x, np.ones(W) / W, "valid")
Q, log = q_train()
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
ax[0].plot(ma(log["ret"])); ax[0].set(title=f"Training return (MA {W})", xlabel="episode")
ax[1].plot(ma(log["success"].astype(float))); ax[1].set(title=f"Training success rate (MA {W})", xlabel="episode", ylim=(0, 1.05))
ax[2].plot(log["eps"]); ax[2].set(title="epsilon", xlabel="episode")
plt.tight_layout(); plt.savefig("q_training.png", dpi=110)
ev = eval1(Q); print("greedy path:", ev["path"], ev)
rb = random_baseline(); ok = [r for r in rb if r["success"]]
rnd = dict(success=len(ok) / 100, timeout=sum(r["timeout"] for r in rb) / 100,
           mean_steps=float(np.mean([r["steps"] for r in ok])) if ok else None,
           mean_invalid=float(np.mean([r["invalid"] for r in rb])))
print("random:", rnd)
print("train: last-50 success %.2f, last-50 return %.2f; first-50 success %.2f" % (log["success"][-50:].mean(), log["ret"][-50:].mean(), log["success"][:50].mean()))
print("first episode with success:", int(np.argmax(log["success"])), " total train timeouts:", int(log["timeout"].sum()))
print("Q at start row (2,0..4):\n", np.round(Q[[idx((2, c)) for c in range(5)]], 2))
# optional: no exploration (eps=0), seeds 7,8,9, random tie-break, zero init
for sd in (7, 8, 9):
    Q0, l0 = q_train(seed=sd, eps0=0.0, eps1=0.0)
    e0 = eval1(Q0); print(f"eps=0 seed {sd}: train success {l0['success'].mean():.2f} greedy success={e0['success']} steps={e0['steps']} path={e0['path']}")
    Qe, le = q_train(seed=sd); ee = eval1(Qe); print(f"eps-greedy seed {sd}: train success {le['success'].mean():.2f} greedy success={ee['success']} steps={ee['steps']}")
e2 = eval1(Q, (0, 0)); print("start (0,0):", e2)
fig, ax = plt.subplots(figsize=(4.5, 4.5))
for r, c in OBST: ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="gray"))
ax.plot(*zip(*[(c, r) for r, c in ev["path"]]), "o-", c="C0", ms=9)
for i, (r, c) in enumerate(ev["path"]): ax.text(c, r - .25, str(i), ha="center", fontsize=9)
ax.text(0, 2.35, "S", ha="center", c="g", weight="bold"); ax.text(4, 2.35, "G", ha="center", c="r", weight="bold")
ax.set(xlim=(-.5, 4.5), ylim=(4.5, -.5), xticks=range(5), yticks=range(5), title="Greedy Q-table trajectory"); ax.grid(True)
plt.savefig("q_traj.png", dpi=110)
np.save("Q.npy", Q)
json.dump(dict(greedy=ev, random=rnd, start00=e2), open("res1.json", "w"), default=str)
