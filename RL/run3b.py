import sys, json, pickle, numpy as np
from lab import *
steps = int(sys.argv[1]); env = Env2(); out = {}
for sd in map(int, sys.argv[2:]):
    net, h = dqn_train(env, steps, seed=sd); ev = eval_net(env, net); ex = eval_net(env, net, ((0, 0), (4, 4)))
    print(f"{steps} seed {sd}: success={ev['success']} coll={ev['collision']} timeout={ev['timeout']} steps={ev['steps']} invalid={ev['invalid']} | alt success={ex['success']} steps={ex['steps']} invalid={ex['invalid']}", flush=True)
    out[sd] = dict(ev=ev, ex=ex)
    pickle.dump((net.W, net.b), open(f"net2_{steps}_{sd}.pkl", "wb"))
json.dump(out, open(f"res3b_{steps}.json", "w"), default=str)
