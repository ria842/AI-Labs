import json, time, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lab import *
t0 = time.time(); env = Env1()
net, h = dqn_train(env, 12000, seed=7)
print(f"time {time.time()-t0:.1f}s updates={h['updates']} episodes={len(h['ep_ret'])}")
ev = eval_net(env, net); print("DQN greedy:", ev)
e2 = eval_net(env, net, (0, 0)); print("DQN from (0,0):", {k: e2[k] for k in ("path", "success", "invalid", "timeout")})
r = np.array(h["ep_ret"]); print("last-50 train return %.2f, first-50 %.2f" % (r[-50:].mean(), r[:50].mean()))
# greedy Q-values along route
for p in ev["path"]: print(p, np.round(net.q(env.enc(p)[None])[0], 2))
# other seeds for robustness
for sd in (8, 9):
    n2, _ = dqn_train(env, 12000, seed=sd); e = eval_net(env, n2); print("seed", sd, e["success"], e["steps"], e["invalid"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ma = lambda x, w=50: np.convolve(x, np.ones(w) / w, "valid")
ax[0].plot(ma(r)); ax[0].set(title="DQN training return (MA 50 episodes)", xlabel="episode")
ax[1].plot(h["loss"]); ax[1].set(title="TD loss", xlabel="optimizer update", yscale="log")
plt.tight_layout(); plt.savefig("dqn1_training.png", dpi=110)
json.dump(dict(greedy=ev, start00=e2), open("res2.json", "w"), default=str)
