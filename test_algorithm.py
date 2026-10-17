"""Numerical oracles for Double DQN targets, terminal masking and Polyak updates."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
import dqn_agent

class DoubleDQNOracle(unittest.TestCase):
    def test_online_selects_target_evaluates_and_terminal_masks(self):
        agent = dqn_agent.Agent(2, 2, 7)
        agent.qnetwork_local = nn.Linear(2, 2, bias=False).to(dqn_agent.device)
        agent.qnetwork_target = nn.Linear(2, 2, bias=False).to(dqn_agent.device)
        with torch.no_grad():
            agent.qnetwork_local.weight.copy_(torch.tensor([[3., 1.], [1., 4.]], device=dqn_agent.device))
            agent.qnetwork_target.weight.copy_(torch.tensor([[2., 9.], [8., 1.]], device=dqn_agent.device))
        # Deliberately disagreeing action rankings distinguish Double DQN from DQN.
        agent.optimizer = torch.optim.SGD(agent.qnetwork_local.parameters(), lr=0.)
        states = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], device=dqn_agent.device)
        next_states = torch.tensor([[1., 0.], [0., 1.], [0., 1.]], device=dqn_agent.device)
        actions = torch.zeros((3, 1), dtype=torch.long, device=dqn_agent.device)
        rewards = torch.tensor([[1.], [7.], [0.]], device=dqn_agent.device)
        terminal = torch.tensor([[0.], [1.], [0.]], device=dqn_agent.device)
        observed = []
        original_loss = dqn_agent.F.mse_loss
        def capture(prediction, target):
            observed.append(target.detach().clone())
            self.assertFalse(target.requires_grad)
            return original_loss(prediction, target)
        with patch.object(dqn_agent.F, 'mse_loss', side_effect=capture):
            agent.learn((states, actions, rewards, next_states, terminal), gamma=.5)
        # Hand-calculated: 1+.5*2, terminal reward7, 0+.5*1.
        torch.testing.assert_close(observed[0].cpu(), torch.tensor([[2.], [7.], [.5]]))
        self.assertTrue(all(p.grad is None for p in agent.qnetwork_target.parameters()))

    def test_polyak_update_is_numerically_correct(self):
        agent = dqn_agent.Agent(2, 2, 7)
        local, target = nn.Linear(2, 1), nn.Linear(2, 1)
        with torch.no_grad():
            local.weight.fill_(6); local.bias.fill_(2)
            target.weight.fill_(2); target.bias.fill_(-2)
        agent._soft_update(local, target, .25)
        torch.testing.assert_close(target.weight, torch.full_like(target.weight, 3.))
        torch.testing.assert_close(target.bias, torch.full_like(target.bias, -1.))

if __name__ == '__main__':
    unittest.main()
