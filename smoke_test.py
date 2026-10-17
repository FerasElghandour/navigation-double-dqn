import numpy as np
import torch
from dqn_agent import Agent,ReplayBuffer
torch.set_num_threads(2)
agent=Agent(8,4,7)
action=agent.act(np.zeros(8,dtype=np.float32),eps=0.)
assert 0<=action<4
replay=ReplayBuffer(4,128,8,7)
state=np.zeros(8)
for i in range(8):replay.add(state,i%4,1.,state,False)
state[:]=999
sample=replay.sample()
assert sample[0].max()==0
before=[p.detach().clone() for p in agent.qnetwork_local.parameters()]
agent.learn(sample,.99)
assert any(not torch.equal(a,b) for a,b in zip(before,agent.qnetwork_local.parameters()))
assert all(torch.isfinite(p).all() for p in agent.qnetwork_local.parameters())
assert all(p.grad is None for p in agent.qnetwork_target.parameters())
print('Double DQN: greedy action, replay ownership, finite parameter updates, and detached targets passed.')
