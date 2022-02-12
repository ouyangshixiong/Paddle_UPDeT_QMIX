import paddle.nn as nn
import paddle.nn.functional as F
import paddle
import numpy as np
import argparse

from paddle.nn import Transformer


class UPDeT(nn.Layer):
    def __init__(self, input_shape, config):
        super(UPDeT, self).__init__()
        self.config = config
        self.transformer = Transformer(self.config['token_dim'], self.config['emb'], self.config['heads'], self.config['depth'], self.config['emb'])
        self.q_basic = nn.Linear(self.config['emb'], 6)

    def init_hidden(self):
        # make hidden states on same device as model
        # return torch.zeros(1, self.args.emb).cuda()
        return np.zeros((1, self.config['emb']))

    def forward(self, inputs, hidden_state, task_enemy_num, task_ally_num):
        outputs, _ = self.transformer.forward(inputs, hidden_state, None)
        # first output for 6 action (no_op stop up down left right)
        q_basic_actions = self.q_basic(outputs[:, 0, :])

        # last dim for hidden state
        h = outputs[:, -1:, :]

        q_enemies_list = []

        # each enemy has an output Q
        for i in range(task_enemy_num):
            q_enemy = self.q_basic(outputs[:, 1 + i, :])
            q_enemy_mean = np.mean(q_enemy, 1, True)
            q_enemies_list.append(q_enemy_mean)

        # concat enemy Q over all enemies
        q_enemies = np.stack(q_enemies_list, dim=1).squeeze()

        # concat basic action Q with enemy attack Q
        q = np.cat((q_basic_actions, q_enemies), 1)

        return q, h


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unit Testing')
    parser.add_argument('--token_dim', default='5', type=int)
    parser.add_argument('--emb', default='32', type=int)
    parser.add_argument('--heads', default='3', type=int)
    parser.add_argument('--depth', default='2', type=int)
    parser.add_argument('--ally_num', default='5', type=int)
    parser.add_argument('--enemy_num', default='5', type=int)
    parser.add_argument('--episode', default='20', type=int)
    args = parser.parse_args()


    # testing the agent
    agent = UPDeT(None, args).cuda()
    hidden_state = agent.init_hidden().cuda().expand(args.ally_num, 1, -1)
    tensor = np.rand(args.ally_num, args.ally_num+args.enemy_num, args.token_dim).cuda()
    q_list = []
    for _ in range(args.episode):
        q, hidden_state = agent.forward(tensor, hidden_state, args.ally_num, args.enemy_num)
        q_list.append(q)
