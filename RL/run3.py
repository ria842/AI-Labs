import json, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lab import *
env = Env2(); res = {}
for sd in (7, 8, 9, 10):
    net, h = dqn_train(env, 60000, seed=sd)
    ev = eval_net(env, net); ex = eval_net(env, net, ((0, 0), (4, 4)))
    ends = np.array(h["ep_end"]); er = np.array(h["ep_ret"])
    print(f"seed {sd}: eps={len(er)} last-100 train ret {er[-100:].mean():.2f} | greedy success={ev['success']} coll={ev['collision']} timeout={ev['timeout']} steps={ev['steps']} invalid={ev['invalid']}")
    print("   path:", ev["path"]); print("   alt start:", {k: ex[k] for k in ("success", "collision", "timeout", "steps", "invalid")}, ex["path"][-1])
    res[sd] = dict(ev=ev, ex=ex, last100=float(er[-100:].mean()))
    if sd == 7: net7, h7 = net, h
json.dump(res, open("res3.json", "w"), default=str)
import pickle; pickle.dump((net7.W, net7.b), open("net2_seed7.pkl", "wb"))
