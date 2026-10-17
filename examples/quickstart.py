"""Offline example using the public implementation; see README for evidence scope."""
import sys,json,pathlib,os
repo=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'src'));os.chdir(repo)
import numpy as np,torch
from dqn_agent import Agent,ReplayBuffer
torch.set_num_threads(2)
agent=Agent(8,4,7); action=agent.act(np.zeros(8,dtype=np.float32),eps=0.)
replay=ReplayBuffer(4,128,8,7)
for i in range(8): replay.add(np.zeros(8),i%4,1.,np.zeros(8),False)
before=[p.detach().clone() for p in agent.qnetwork_local.parameters()]
agent.learn(replay.sample(),.99)
print(json.dumps({"fixture":"untrained Q network/generated transitions; no environment score","action":action,"replay_size":len(replay),"online_changed":any(not torch.equal(a,b) for a,b in zip(before,agent.qnetwork_local.parameters()))}))
