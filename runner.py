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

import time
import parl
import numpy as np

from smac.env import StarCraft2Env
from env_wrapper import SC2EnvWrapper
from copy import deepcopy
from components.episode_buffer import ReplayBuffer

from transformer_model import TransformerModel
from qmixer_model import QMixerModel
from qmix import QMIX
from qmix_agent import QMixAgent
from actor import Actor
from parl.utils import logger
from functools import partial
from components.episode_buffer import EpisodeBatch

class Runner(object):
    def __init__(self, config, scheme, groups, preprocess):
        self.config = config
        #=== Create Agent ===
        self.__create_agent()

        #=== init Runner params ===
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)

        #=== Remote Actor ===
        parl.connect(self.config['master_address'])
        self.__create_actors()

    def __create_agent(self):
        # init config from env
        env = StarCraft2Env(map_name=self.config['scenario'], difficulty=self.config['difficulty'])
        env = SC2EnvWrapper(env)
        self.config['episode_limit'] = env.episode_limit
        self.config['obs_shape'] = env.obs_shape
        self.config['state_shape'] = env.state_shape
        self.config['n_agents'] = env.n_agents
        self.config['n_actions'] = env.n_actions
        
        # 从RNN改成Transformer
        agent_model = TransformerModel(self.config['obs_shape'], self.config['n_actions'],
                            self.config['rnn_hidden_dim'])
        qmixer_model = QMixerModel(
                    self.config['n_agents'], self.config['state_shape'], self.config['mixing_embed_dim'],
                    self.config['hypernet_layers'], self.config['hypernet_embed_dim'])
        algorithm = QMIX(agent_model, qmixer_model, self.config['double_q'],
                    self.config['gamma'], self.config['lr'], self.config['clip_grad_norm'])
        self.qmix_agent = QMixAgent(
                    algorithm, self.config['exploration_start'], self.config['min_exploration'],
                    self.config['exploration_decay'], self.config['update_target_interval'])

    def __create_actors(self):
        self.remote_actors = [
            Actor(self.config) for _ in range(self.config['actor_num'])
        ]
        logger.info('Creating {} remote actors to connect.'.format(
            self.config['actor_num']))

    def run(self):
        self.reset()

        # should be True + break
        while False:
            pass
            # select action

            # send action to actor

            # receive data back

        return self.batch

    def reset(self):
        self.batch = self.new_batch()

        # reset envs

