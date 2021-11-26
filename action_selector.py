#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from updet_framework import UPDeT


class ActionSelector:
    def __init__(self, scheme, config):
        self.config = config
        self.n_agents = config['n_agents']
        
        input_shape = self._get_input_shape(scheme)
        self.updet = UPDeT(input_shape, self.config)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None)):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self._apply_transformer(ep_batch, t_ep)
        chosen_actions = self._epsilon_greedy(agent_outputs[bs], avail_actions[bs], t_env)
        return chosen_actions

    def init_hidden(self, batch_size):
        self.hidden_states = self.updet.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, 1, -1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.config['obs_last_action']:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.config['obs_agent_id']:
            input_shape += self.n_agents
        return input_shape

    def _apply_transformer(self, ep_batch, t):
        agent_inputs = self._build_inputs_transformer(ep_batch, t)
        agent_outs, self.hidden_states = self.updet(agent_inputs,
                                                           self.hidden_states.reshape(-1, 1, self.config['emb']),
                                                           self.config['enemy_num'], self.config['ally_num'])

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)


    def _build_inputs_transformer(self, batch, t):
        # currently we only support battles with marines (e.g. 3m 8m 5m_vs_6m)
        # you can implement your own with any other agent type.
        inputs = []
        raw_obs = batch["obs"][:, t]
        arranged_obs = np.concatenate((raw_obs[:, :, -1:], raw_obs[:, :, :-1]), 2)
        reshaped_obs = arranged_obs.reshape(-1, 1 + (self.config['enemy_num'] - 1) + self.config['ally_num'], self.config['token_dim'])
        inputs.append(reshaped_obs)
        #inputs = th.cat(inputs, dim=1).cuda()
        inputs = np.concatenate(inputs, axis=1)
        return inputs

    def _epsilon_greedy(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions